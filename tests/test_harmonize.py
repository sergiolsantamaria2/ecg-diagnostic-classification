"""Lead reordering into the canonical 12-lead layout.

Resampling and length fitting are covered through the serving preprocessor in
``test_preprocess.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from deep_ecg.data.harmonize import reorder_to_canonical
from deep_ecg.data.sources.base import CANONICAL_LEADS


def test_reorder_handles_shuffled_lowercase_names():
    canonical = np.tile(np.arange(12, dtype=np.float32), (5, 1))  # column j == lead j
    perm = np.random.default_rng(0).permutation(12)
    shuffled = canonical[:, perm]
    names = [CANONICAL_LEADS[i].lower() for i in perm]
    np.testing.assert_array_equal(reorder_to_canonical(shuffled, names), canonical)


def test_reorder_missing_lead_raises():
    with pytest.raises(KeyError):
        reorder_to_canonical(np.zeros((5, 11), np.float32), list(CANONICAL_LEADS[:11]))
