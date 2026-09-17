"""Multi-label metrics and threshold tuning."""

from __future__ import annotations

import numpy as np

from deep_ecg.evaluation.analysis import per_class_report, tune_thresholds
from deep_ecg.evaluation.metrics import compute_metrics


def test_perfect_scores_give_unit_auroc_and_f1():
    y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
    m = compute_metrics(y, y.astype(float), ("A", "B", "C"))
    assert m["macro_auroc"] == 1.0
    assert m["macro_f1"] == 1.0


def test_single_valued_class_is_nan_and_excluded_from_macro():
    y = np.array([[1, 1], [0, 1], [1, 1]])
    s = np.array([[0.9, 0.2], [0.1, 0.3], [0.8, 0.4]])
    m = compute_metrics(y, s, ("A", "B"))
    assert np.isnan(m["per_class_auroc"]["B"])
    assert m["macro_auroc"] == 1.0


def test_tune_thresholds_picks_a_separating_value():
    y = np.array([[1], [1], [0], [0]])
    s = np.array([[0.9], [0.7], [0.3], [0.1]])
    thresholds = tune_thresholds(y, s, ("A",))
    assert set(thresholds) == {"A"}
    assert 0.3 < thresholds["A"] <= 0.7
    row = per_class_report(y, s, ("A",), thresholds)[0]
    assert (row["tp"], row["fp"], row["fn"]) == (2, 0, 0)
