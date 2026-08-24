"""Input formatting, tokenisation and batching for ALICE.

Two input encodings are supported, matching the paper:

``structured``
    XML-style segment tags, used for LLM-based (decoder) encoders. This is the
    better-performing of the two formats reported by Ruan et al. (2024).
    Tags are German, matching the language of the data::

        <Frage>...</Frage>
        <Loesung>...</Loesung>
        <Antwort>...</Antwort>
        <Rubrik>...</Rubrik>

``sep``
    Segments joined with the tokeniser's ``[SEP]`` token, used for masked
    language models.

Segments are always emitted in the order question, sample solution, answer,
rubric, regardless of the order in which they are requested; ``--input-fields``
selects *which* segments appear, not their order.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch

__all__ = [
    "FIELD_ALIASES",
    "FIELD_TAGS",
    "FIELD_ORDER",
    "parse_input_fields",
    "get_tokenizer",
    "render_input",
    "encode_rubric_candidates",
    "encode_sequence_classification",
    "rubric_retrieval_collate",
    "sequence_classification_collate",
]

#: Short flag letters used on the command line.
FIELD_ALIASES: dict[str, str] = {
    "q": "question",
    "s": "sample_solution",
    "a": "answer",
    "r": "rubric",
}

#: German segment tags used by the ``structured`` format.
FIELD_TAGS: dict[str, str] = {
    "question": "Frage",
    "sample_solution": "Loesung",
    "answer": "Antwort",
    "rubric": "Rubrik",
}

#: Canonical rendering order.
FIELD_ORDER: tuple[str, ...] = ("question", "sample_solution", "answer", "rubric")

MAX_LENGTH = 512


def parse_input_fields(spec: str) -> list[str]:
    """Expand a flag like ``"qsar"`` or ``"q,s,a,r"`` into full field names."""
    letters = [c for c in spec.replace(",", "").replace("+", "").strip() if not c.isspace()]
    unknown = sorted(set(letters) - set(FIELD_ALIASES))
    if unknown:
        raise ValueError(f"Unknown input field(s) {unknown}; valid letters: {sorted(FIELD_ALIASES)}")
    selected = {FIELD_ALIASES[c] for c in letters}
    if "answer" not in selected:
        raise ValueError("Input fields must include the student answer ('a')")
    return [field for field in FIELD_ORDER if field in selected]


def get_tokenizer(base_model: str):
    """Load a tokeniser and make sure padding/separator tokens are usable."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    if tokenizer.sep_token is None:
        tokenizer.sep_token = tokenizer.eos_token
    return tokenizer


def _values(instance: dict[str, Any], fields: Sequence[str], rubric: str | None) -> list[tuple[str, str]]:
    out = []
    for field in fields:
        value = rubric if field == "rubric" else instance.get(field, "")
        if field == "rubric" and value is None:
            continue
        out.append((field, str(value).strip()))
    return out


def render_input(
    instance: dict[str, Any],
    fields: Sequence[str],
    rubric: str | None = None,
    rubrics: Sequence[str] | None = None,
    input_format: str = "structured",
    sep_token: str = "[SEP]",
) -> str:
    """Render one model input string.

    Args:
        instance: An :class:`~data_utils.data_loader.Instance` as a dict.
        fields: Full field names, e.g. ``["question", "answer", "rubric"]``.
        rubric: A single rubric candidate, for rubric-retrieval.
        rubrics: The full rubric set, for sequence classification. Ignored when
            ``rubric`` is given.
    """
    if rubric is None and rubrics is not None and "rubric" in fields:
        segments = _values(instance, [f for f in fields if f != "rubric"], None)
        rubric_values = [("rubric", str(r).strip()) for r in rubrics]
        segments = segments + rubric_values
    else:
        segments = _values(instance, fields, rubric)

    if input_format == "structured":
        return "\n".join(f"<{FIELD_TAGS[f]}>{v}</{FIELD_TAGS[f]}>" for f, v in segments)
    if input_format == "sep":
        return sep_token.join(v for _, v in segments)
    raise ValueError(f"Unknown input_format {input_format!r}; use 'structured' or 'sep'")


def encode_rubric_candidates(
    instance: dict[str, Any],
    tokenizer,
    fields: Sequence[str],
    input_format: str = "structured",
    max_length: int = MAX_LENGTH,
) -> dict[str, Any]:
    """Tokenise one instance into ``num_rubrics`` candidate sequences.

    Returns ``input_ids``/``attention_mask`` of shape ``[R, S]`` plus the index
    of the gold rubric in ``labels``.
    """
    fields = list(fields)
    if "rubric" not in fields:
        raise ValueError("Rubric-retrieval requires the rubric field ('r')")

    texts = [
        render_input(
            instance,
            fields,
            rubric=rubric,
            input_format=input_format,
            sep_token=tokenizer.sep_token,
        )
        for rubric in instance["rubrics"]
    ]
    encoded = tokenizer(texts, truncation=True, max_length=max_length)
    return {
        "id": instance["id"],
        "question_id": instance["question_id"],
        "subject": instance["subject"],
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "num_rubrics": len(texts),
        "labels": int(instance["level"]),
        "global_level": int(instance["global_level"]),
        # Maps a predicted rubric position back onto the global level scheme,
        # which matters for questions that omit intermediate levels.
        "level_names": list(instance["level_names"]),
    }


def encode_sequence_classification(
    instance: dict[str, Any],
    tokenizer,
    fields: Sequence[str],
    input_format: str = "structured",
    max_length: int = MAX_LENGTH,
) -> dict[str, Any]:
    """Tokenise one instance into a single sequence with a fixed-head label.

    The label is the *global* level, not the within-question rubric position:
    the classification head assumes one label space shared across questions.
    For ``ke``/``sk`` the element/skill name is part of the rendered rubric
    text, so that the head knows which item is being scored.
    """
    fields = list(fields)
    item = instance.get("item") or ""
    payload = dict(instance)
    if item and "rubric" not in fields:
        # Without rubric segments the item name would be lost entirely.
        payload["answer"] = f"{item}: {instance['answer']}"

    text = render_input(
        payload,
        fields,
        rubrics=instance["rubrics"],
        input_format=input_format,
        sep_token=tokenizer.sep_token,
    )
    encoded = tokenizer(text, truncation=True, max_length=max_length)
    return {
        "id": instance["id"],
        "question_id": instance["question_id"],
        "subject": instance["subject"],
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": int(instance["global_level"]),
        "level": int(instance["level"]),
        "num_rubrics": int(instance["num_rubrics"]),
    }


def _meta(batch: Iterable[dict[str, Any]], keys: Sequence[str]) -> dict[str, list[Any]]:
    batch = list(batch)
    return {key: [example.get(key) for example in batch] for key in keys}


def rubric_retrieval_collate(
    batch: list[dict[str, Any]], pad_id: int = 0, return_meta: bool = False
):
    """Pad a batch of candidate sets to ``[B, R, S]``.

    Both the rubric dimension ``R`` and the sequence dimension ``S`` are ragged:
    questions define different numbers of levels and the candidate texts differ
    in length. Padded rubric slots are flagged in ``rubric_mask`` so the loss
    and the arg-max ignore them.
    """
    batch_size = len(batch)
    max_rubrics = max(example["num_rubrics"] for example in batch)
    max_length = max(len(ids) for example in batch for ids in example["input_ids"])

    input_ids = torch.full((batch_size, max_rubrics, max_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_rubrics, max_length), dtype=torch.long)
    rubric_mask = torch.zeros((batch_size, max_rubrics), dtype=torch.bool)

    for i, example in enumerate(batch):
        for j, (ids, mask) in enumerate(zip(example["input_ids"], example["attention_mask"])):
            input_ids[i, j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, j, : len(mask)] = torch.tensor(mask, dtype=torch.long)
        rubric_mask[i, : example["num_rubrics"]] = True

    collated = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "rubric_mask": rubric_mask,
        "labels": torch.tensor([example["labels"] for example in batch], dtype=torch.long),
    }
    if return_meta:
        return collated, _meta(
            batch, ["id", "question_id", "subject", "global_level", "num_rubrics", "level_names"]
        )
    return collated


def sequence_classification_collate(
    batch: list[dict[str, Any]], pad_id: int = 0, return_meta: bool = False
):
    """Pad a batch of single sequences to ``[B, S]``."""
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch],
        batch_first=True,
        padding_value=pad_id,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(example["attention_mask"], dtype=torch.long) for example in batch],
        batch_first=True,
        padding_value=0,
    )
    collated = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor([example["labels"] for example in batch], dtype=torch.long),
    }
    if return_meta:
        return collated, _meta(batch, ["id", "question_id", "subject", "level", "num_rubrics"])
    return collated
