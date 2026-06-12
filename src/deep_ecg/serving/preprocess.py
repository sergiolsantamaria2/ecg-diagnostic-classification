"""Inference-time preprocessing for a raw 12-lead ECG.

A request carries a ``(12, L)`` signal at some sampling rate; the model was
trained on a fixed canonical format (12 leads, 100 Hz, 1000 samples) and on
per-lead z-scored inputs. This applies the exact same harmonization and
standardization used during training — the shared :mod:`deep_ecg.data.harmonize`
helpers and the training :class:`~deep_ecg.data.transforms.Standardize` — so the
signal the network sees in production matches the one it saw while learning.
"""

from __future__ import annotations

import numpy as np

from ..data.harmonize import fit_length, resample_to_rate
from ..data.sources.base import CANONICAL_LEADS
from ..data.transforms import Standardize

N_LEADS = len(CANONICAL_LEADS)


class Preprocessor:
    """Map a raw ``(12, L)`` ECG to the standardized ``(12, target_len)`` input.

    Leads are assumed to already be in canonical order (:data:`CANONICAL_LEADS`);
    the request schema documents and validates this. The signal is resampled to
    ``target_fs`` Hz, center-cropped or zero-padded to ``target_len`` samples, and
    z-scored with the training lead statistics.
    """

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        target_fs: int = 100,
        target_len: int = 1000,
    ) -> None:
        self.standardize = Standardize(mean, std)
        self.target_fs = target_fs
        self.target_len = target_len

    def __call__(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Return the standardized ``(12, target_len)`` float32 signal."""
        x = np.asarray(signal, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] != N_LEADS:
            raise ValueError(f"expected a (12, L) signal, got shape {x.shape}")
        if x.shape[1] == 0:
            raise ValueError("signal has no samples")
        if not np.isfinite(x).all():
            raise ValueError("signal contains non-finite values")
        x = resample_to_rate(x, int(sampling_rate), self.target_fs)
        x = fit_length(x, self.target_len)
        return self.standardize(x).astype(np.float32)
