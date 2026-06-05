"""Multi-label evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Sequence[str],
    threshold: float = 0.5,
) -> dict:
    """Compute macro-AUROC (primary), per-class AUROC and F1.

    ``y_true`` is a ``(N, C)`` binary matrix and ``y_score`` a ``(N, C)`` matrix
    of predicted probabilities. AUROC is undefined for a class with a single
    label value present, which is reported as NaN and excluded from the macro
    average.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    per_class_auroc: dict[str, float] = {}
    for i, name in enumerate(class_names):
        if np.unique(y_true[:, i]).size < 2:
            per_class_auroc[name] = float("nan")
        else:
            per_class_auroc[name] = float(roc_auc_score(y_true[:, i], y_score[:, i]))
    macro_auroc = float(np.nanmean(list(per_class_auroc.values())))

    y_pred = (y_score >= threshold).astype(int)
    per_class_f1 = {
        name: float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
        for i, name in enumerate(class_names)
    }
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "macro_auroc": macro_auroc,
        "macro_f1": macro_f1,
        "per_class_auroc": per_class_auroc,
        "per_class_f1": per_class_f1,
    }
