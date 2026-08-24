"""Wiring shared by training and inference.

Two modelling approaches are supported, both described in the paper:

``rubric-retrieval``
    Encode the answer jointly with each candidate rubric level and rank the
    candidates with a softmax over the question's own rubric set
    (:class:`~alice.models.RubricRetrievalModel`).

``seq-classification``
    Concatenate the answer with the sample solution (and optionally the
    question and rubrics) and predict a level with a fixed classification head
    over the dataset-wide level scheme.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from alice.data import GLOBAL_NUM_LEVELS, Instance
from alice.encoding import (
    encode_rubric_candidates,
    encode_sequence_classification,
    parse_input_fields,
    rubric_retrieval_collate,
    sequence_classification_collate,
)

__all__ = [
    "APPROACHES",
    "ModelConfig",
    "encode_split",
    "build_collate_fn",
    "build_model",
    "predict",
]

APPROACHES: tuple[str, ...] = ("rubric-retrieval", "seq-classification")

#: Substrings that identify a causal-LM encoder, which needs LoRA-friendly
#: defaults (last-token pooling, right padding, paged optimiser).
CAUSAL_LM_HINTS = ("llama", "mistral", "qwen", "gemma", "phi", "olmo", "gpt")


def is_causal_lm(base_model: str) -> bool:
    name = base_model.lower()
    return any(hint in name for hint in CAUSAL_LM_HINTS)


@dataclass
class ModelConfig:
    """Everything needed to build a model and its inputs."""

    base_model: str = "jhu-clsp/mmBERT-base"
    approach: str = "rubric-retrieval"
    task: str = "lp"
    input_fields: str = "ar"
    input_format: str = "auto"
    pool_type: str = "auto"
    max_length: int = 512
    use_lora: bool = False
    use_4bit: bool = False
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.approach not in APPROACHES:
            raise ValueError(f"approach must be one of {APPROACHES}, got {self.approach!r}")
        self.fields: list[str] = parse_input_fields(self.input_fields)
        if self.input_format == "auto":
            self.input_format = "structured" if is_causal_lm(self.base_model) else "sep"
        if self.pool_type == "auto":
            self.pool_type = "last" if is_causal_lm(self.base_model) else "avg"

    @property
    def num_labels(self) -> int:
        return GLOBAL_NUM_LEVELS[self.task]

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }


def encode_split(
    instances: Sequence[Instance],
    tokenizer,
    config: ModelConfig,
):
    """Tokenise one split into a ``datasets.Dataset``."""
    from datasets import Dataset

    encode = (
        encode_rubric_candidates
        if config.approach == "rubric-retrieval"
        else encode_sequence_classification
    )
    rows = [
        encode(
            instance.to_dict(),
            tokenizer,
            config.fields,
            input_format=config.input_format,
            max_length=config.max_length,
        )
        for instance in instances
    ]
    return Dataset.from_list(rows)


def build_collate_fn(config: ModelConfig, tokenizer) -> Callable:
    base = (
        rubric_retrieval_collate
        if config.approach == "rubric-retrieval"
        else sequence_classification_collate
    )
    return partial(base, pad_id=tokenizer.pad_token_id or 0)


def _quantization_config(config: ModelConfig):
    if not (config.use_4bit and torch.cuda.is_available()):
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _lora_config(config: ModelConfig):
    if not config.use_lora:
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="SEQ_CLS" if config.approach == "seq-classification" else None,
    )


def build_model(
    config: ModelConfig,
    checkpoint: str | None = None,
    pad_token_id: int | None = None,
):
    """Build a fresh model, or load one from ``checkpoint``.

    ``pad_token_id`` is only consulted by the sequence-classification head,
    which pools at the last non-padding position and therefore has to know
    which token is padding. Causal LMs usually ship without a pad token, so
    pass the tokeniser's.
    """
    from transformers import AutoConfig, AutoModelForSequenceClassification

    from alice.models import RubricRetrievalModel

    quantization_config = _quantization_config(config)
    lora_config = _lora_config(config)

    if config.approach == "rubric-retrieval":
        source = checkpoint or config.base_model
        model_config = AutoConfig.from_pretrained(source)
        model_config.base_model_name_or_path = config.base_model
        model_config.pool_type = config.pool_type
        if checkpoint:
            return RubricRetrievalModel.from_pretrained(
                checkpoint,
                config=model_config,
                quantization_config=quantization_config,
            )
        return RubricRetrievalModel(
            model_config,
            lora_config=lora_config,
            quantization_config=quantization_config,
        )

    if checkpoint:
        from peft import PeftModelForSequenceClassification
        import os

        model_config = AutoConfig.from_pretrained(checkpoint)
        has_adapter = os.path.exists(os.path.join(checkpoint, "adapter_config.json"))
        if has_adapter:
            base = AutoModelForSequenceClassification.from_pretrained(
                config.base_model,
                config=model_config,
                quantization_config=quantization_config,
            )
            return PeftModelForSequenceClassification.from_pretrained(base, checkpoint)
        return AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            config=model_config,
            quantization_config=quantization_config,
        )

    model_config = AutoConfig.from_pretrained(config.base_model)
    model_config.num_labels = config.num_labels
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        config=model_config,
        quantization_config=quantization_config,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = pad_token_id if pad_token_id is not None else model_config.eos_token_id
    if lora_config is not None:
        from peft import get_peft_model

        model = get_peft_model(model, lora_config)
    return model


@torch.no_grad()
def predict(
    model,
    dataset,
    collate_fn: Callable,
    batch_size: int = 16,
    device: torch.device | None = None,
    desc: str = "predicting",
) -> tuple[list[dict[str, Any]], float]:
    """Run inference and return per-instance rows plus wall-clock seconds.

    Rows carry the model's prediction in both indexings where they differ:
    ``pred`` on whatever scale the model outputs, and ``pred_global`` on the
    dataset-wide level scheme so that predictions from the two approaches are
    directly comparable.
    """
    device = device or next(model.parameters()).device
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, return_meta=True),
    )

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch, meta in tqdm(loader, desc=desc):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
        predictions = logits.argmax(dim=-1).cpu().tolist()
        gold = batch["labels"].cpu().tolist()

        for i, (pred, label) in enumerate(zip(predictions, gold)):
            row = {
                "id": meta["id"][i],
                "question_id": meta["question_id"][i],
                "subject": meta["subject"][i],
                "gold": label,
                "pred": pred,
            }
            if "level_names" in meta:
                # rubric-retrieval predicts a position inside the question's own
                # rubric list, so map it back onto the global level scheme.
                level_names = meta["level_names"][i]
                row["gold_global"] = meta["global_level"][i]
                row["pred_global"] = int(float(level_names[pred]))
            else:
                # seq-classification: labels already live on the global scheme
                row["gold_global"] = label
                row["pred_global"] = pred
            rows.append(row)
    elapsed = time.perf_counter() - started
    return rows, elapsed
