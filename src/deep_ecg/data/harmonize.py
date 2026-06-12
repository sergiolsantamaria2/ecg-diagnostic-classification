"""Signal harmonization to the canonical ECG format.

Every database — and every request served in production — is brought to the same
representation an encoder expects: the 12 standard leads in canonical order, a
fixed sampling rate, and a fixed number of samples. These pure functions are the
single implementation of that mapping, shared by the source adapters and the
serving preprocessor so training and inference cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from .sources.base import CANONICAL_LEADS


def reorder_to_canonical(signal: np.ndarray, sig_names: Sequence[str]) -> np.ndarray:
    """Reorder a ``(T, n)`` signal's columns into canonical 12-lead order.

    ``sig_names`` are the lead names as stored in the record; the result has its
    columns permuted to :data:`CANONICAL_LEADS`. Raises ``KeyError`` if a
    canonical lead is missing.
    """
    col = {name.upper(): i for i, name in enumerate(sig_names)}
    order = [col[lead.upper()] for lead in CANONICAL_LEADS]
    return signal[:, order]


def resample_to_rate(signal: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """Polyphase anti-aliased resample of a ``(12, T)`` signal to ``target_fs``."""
    if orig_fs == target_fs:
        return signal.astype(np.float32, copy=False)
    g = gcd(orig_fs, target_fs)
    up, down = target_fs // g, orig_fs // g
    return resample_poly(signal, up, down, axis=1).astype(np.float32)


def fit_length(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Center-crop or zero-pad a ``(12, T)`` signal to ``target_len`` samples."""
    t = signal.shape[1]
    if t == target_len:
        return signal.astype(np.float32, copy=False)
    if t > target_len:
        start = (t - target_len) // 2
        return signal[:, start : start + target_len].astype(np.float32, copy=False)
    out = np.zeros((signal.shape[0], target_len), dtype=np.float32)
    start = (target_len - t) // 2
    out[:, start : start + t] = signal
    return out
