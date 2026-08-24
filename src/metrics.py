"""Scoring metrics for ALICE.

The paper reports macro F1, quadratic weighted kappa (QWK) and accuracy, each
computed over the flat list of (gold, predicted) pairs of a split, with
per-subject breakdowns alongside.

QWK is only meaningful on an ordinal scale, so it is computed on the *global*
level scheme rather than on within-question rubric positions.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

__all__ = ["score", "score_by_subject", "evaluation_report", "save_report"]


def _qwk(gold: Sequence[int], pred: Sequence[int]) -> float:
    """QWK, guarding the degenerate case of a single observed level."""
    labels = sorted(set(gold) | set(pred))
    if len(labels) < 2:
        return float("nan")
    return float(cohen_kappa_score(gold, pred, weights="quadratic", labels=labels))


def score(gold: Sequence[int], pred: Sequence[int]) -> dict[str, float]:
    """Accuracy, macro F1 and QWK for one flat list of predictions."""
    if len(gold) != len(pred):
        raise ValueError(f"gold/pred length mismatch: {len(gold)} vs {len(pred)}")
    if not gold:
        return {"n": 0, "accuracy": float("nan"), "f1_macro": float("nan"), "qwk": float("nan")}
    return {
        "n": len(gold),
        "accuracy": float(accuracy_score(gold, pred)),
        "f1_macro": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "qwk": _qwk(gold, pred),
    }


def score_by_subject(
    gold: Sequence[int], pred: Sequence[int], subjects: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Per-subject breakdown, keyed by subject prefix."""
    grouped: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for g, p, subject in zip(gold, pred, subjects):
        grouped[subject][0].append(g)
        grouped[subject][1].append(p)
    return {subject: score(g, p) for subject, (g, p) in sorted(grouped.items())}


def evaluation_report(
    gold: Sequence[int],
    pred: Sequence[int],
    subjects: Sequence[str] | None = None,
    with_confusion: bool = True,
) -> dict[str, Any]:
    """Overall metrics plus optional per-subject breakdown and confusion matrix."""
    report: dict[str, Any] = {"overall": score(gold, pred)}
    if subjects is not None:
        report["per_subject"] = score_by_subject(gold, pred, subjects)
    if with_confusion and gold:
        labels = sorted(set(gold) | set(pred))
        report["labels"] = labels
        report["confusion_matrix"] = confusion_matrix(gold, pred, labels=labels).tolist()
    return report


def save_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
