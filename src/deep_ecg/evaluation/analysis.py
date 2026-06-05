"""Error analysis for multi-label classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def tune_thresholds(
    y_true: np.ndarray, y_score: np.ndarray, class_names: Sequence[str],
    grid: np.ndarray | None = None,
) -> dict[str, float]:
    """Per-class decision threshold maximizing F1.

    Intended to be fit on the validation set and applied to test, since the
    default 0.5 threshold is rarely optimal under class imbalance.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    thresholds: dict[str, float] = {}
    for i, name in enumerate(class_names):
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            pred = y_score[:, i] >= t
            tp = int(np.sum(pred & (y_true[:, i] == 1)))
            fp = int(np.sum(pred & (y_true[:, i] == 0)))
            fn = int(np.sum(~pred & (y_true[:, i] == 1)))
            denom = 2 * tp + fp + fn
            f1 = 2 * tp / denom if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[name] = best_t
    return thresholds


def per_class_report(
    y_true: np.ndarray, y_score: np.ndarray, class_names: Sequence[str],
    thresholds: Mapping[str, float] | None = None,
) -> list[dict]:
    """Per-class precision/recall/F1 and TP/FP/FN counts at the given thresholds."""
    rows: list[dict] = []
    for i, name in enumerate(class_names):
        t = 0.5 if thresholds is None else thresholds[name]
        pred = y_score[:, i] >= t
        tp = int(np.sum(pred & (y_true[:, i] == 1)))
        fp = int(np.sum(pred & (y_true[:, i] == 0)))
        fn = int(np.sum(~pred & (y_true[:, i] == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "class": name,
            "threshold": round(t, 3),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(y_true[:, i] == 1)),
            "tp": tp, "fp": fp, "fn": fn,
        })
    return rows


def label_cooccurrence(y_true: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    """Symmetric co-occurrence counts between labels (diagonal = support)."""
    y = (y_true > 0).astype(int)
    return y.T @ y
