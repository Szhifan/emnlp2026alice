"""Rubric-retrieval model for ALICE.

Each instance is expanded into its ``R`` candidate rubric levels. Every
candidate is encoded jointly with the answer (and optionally the question and
sample solution), pooled to a vector, and projected to a scalar alignment
score. Scores compete inside the candidate set through a softmax
cross-entropy, so the model learns to *contrast* the correct rubric level
against the alternatives of the same question rather than to judge each level
in isolation:

.. math::

    P(j \\mid i) = \\frac{\\exp(z_{i,j})}{\\sum_k \\exp(z_{i,k})},
    \\qquad \\ell_i = -\\log P(y_i \\mid i)

Padded candidate slots are masked out before the softmax, which lets a batch
mix questions with different numbers of levels.

The sequence-classification baseline uses stock
``AutoModelForSequenceClassification`` and needs no custom module.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

__all__ = ["Pooler", "RubricRetrievalModel"]

MASK_VALUE = -1e4


class Pooler(nn.Module):
    """Pool token states to one vector per sequence.

    ``avg`` suits masked language models; ``last`` (the final non-padding
    token) is the usual choice for causal LLM encoders.
    """

    def __init__(self, pool_type: str = "avg") -> None:
        super().__init__()
        if pool_type not in ("avg", "cls", "last"):
            raise ValueError(f"pool_type must be avg|cls|last, got {pool_type!r}")
        self.pool_type = pool_type

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        masked = hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        if self.pool_type == "cls":
            return masked[:, 0]
        lengths = attention_mask.sum(dim=1)
        if self.pool_type == "avg":
            return masked.sum(dim=1) / lengths.clamp(min=1)[..., None]
        # last non-padding token
        index = (lengths - 1).clamp(min=0)
        return masked[torch.arange(masked.size(0), device=masked.device), index]


class RubricRetrievalModel(PreTrainedModel):
    """Score every rubric candidate of an instance and rank them.

    Args:
        config: A ``transformers`` config carrying two extra attributes,
            ``base_model_name_or_path`` and ``pool_type``.
        lora_config: Optional ``peft.LoraConfig``; when given the encoder is
            wrapped as a PEFT model.
        quantization_config: Optional ``BitsAndBytesConfig`` for QLoRA.
    """

    supports_gradient_checkpointing = True

    def __init__(self, config, lora_config=None, quantization_config=None) -> None:
        super().__init__(config)
        self.pooler = Pooler(getattr(config, "pool_type", "avg"))
        self.encoder = AutoModel.from_pretrained(
            config.base_model_name_or_path,
            config=config,
            quantization_config=quantization_config,
        )
        self.score = nn.Linear(config.hidden_size, 1, bias=False)
        if lora_config is not None:
            self.init_peft(lora_config)

    # ------------------------------------------------------------------
    # PEFT plumbing
    # ------------------------------------------------------------------
    def init_peft(self, lora_config) -> None:
        from peft import get_peft_model

        if not hasattr(self.encoder, "peft_config"):
            self.encoder = get_peft_model(self.encoder, lora_config)

    def load_adapter_weights(self, adapter_dir: str) -> None:
        from peft import PeftModel

        self.encoder = PeftModel.from_pretrained(self.encoder, adapter_dir)

    @property
    def _is_peft(self) -> bool:
        return hasattr(self.encoder, "peft_config")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self) -> None:
        self.encoder.gradient_checkpointing_disable()

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        """Save the model, keeping LoRA adapters and the scoring head apart."""
        import os

        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        if self._is_peft:
            self.encoder.save_pretrained(save_directory)
            head = {k: v for k, v in self.state_dict().items() if not k.startswith("encoder.")}
            torch.save(head, os.path.join(save_directory, "scoring_head.bin"))
        else:
            torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        model_path: str,
        config=None,
        lora_config=None,
        quantization_config=None,
        **kwargs,
    ) -> "RubricRetrievalModel":
        """Load a checkpoint written by :meth:`save_pretrained`."""
        import os

        config = config or AutoConfig.from_pretrained(model_path)
        has_adapter = any(
            os.path.exists(os.path.join(model_path, name))
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        )

        model = cls(config, lora_config=None, quantization_config=quantization_config)
        if has_adapter:
            model.load_adapter_weights(model_path)
            head_file = os.path.join(model_path, "scoring_head.bin")
            if os.path.exists(head_file):
                model.load_state_dict(torch.load(head_file, map_location="cpu"), strict=False)
            else:
                raise FileNotFoundError(f"scoring_head.bin missing in {model_path}")
        else:
            weights = os.path.join(model_path, "pytorch_model.bin")
            if not os.path.exists(weights):
                raise FileNotFoundError(f"pytorch_model.bin missing in {model_path}")
            model.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
        return model

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,  # [B, R, S]
        attention_mask: torch.Tensor,  # [B, R, S]
        rubric_mask: Optional[torch.Tensor] = None,  # [B, R]
        labels: Optional[torch.LongTensor] = None,  # [B]
        tau: float = 1.0,
    ) -> SequenceClassifierOutput:
        batch, num_rubrics, seq_len = input_ids.shape
        if rubric_mask is None:
            rubric_mask = torch.ones(batch, num_rubrics, dtype=torch.bool, device=input_ids.device)

        flat_ids = input_ids.reshape(batch * num_rubrics, seq_len)
        flat_mask = attention_mask.reshape(batch * num_rubrics, seq_len)
        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)

        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = self.pooler(outputs.last_hidden_state, flat_mask)
        logits = self.score(pooled.to(self.score.weight.dtype)).view(batch, num_rubrics)
        logits = logits.masked_fill(~rubric_mask, MASK_VALUE)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits / tau, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)
