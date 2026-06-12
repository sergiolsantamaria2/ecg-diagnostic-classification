"""Unit tests for the inference preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from deep_ecg.serving.preprocess import Preprocessor

IDENTITY_MEAN = np.zeros(12, dtype=np.float32)
IDENTITY_STD = np.ones(12, dtype=np.float32)


def make_preprocessor(mean=IDENTITY_MEAN, std=IDENTITY_STD):
    return Preprocessor(mean, std, target_fs=100, target_len=1000)


def test_output_shape_at_native_rate():
    pre = make_preprocessor()
    out = pre(np.random.randn(12, 1000).astype(np.float32), sampling_rate=100)
    assert out.shape == (12, 1000)
    assert out.dtype == np.float32


def test_resample_then_fit_length():
    # 500 Hz, 10 s -> resampled to 100 Hz, 1000 samples (no crop/pad needed).
    pre = make_preprocessor()
    out = pre(np.random.randn(12, 5000).astype(np.float32), sampling_rate=500)
    assert out.shape == (12, 1000)


def test_long_signal_is_center_cropped():
    # 100 Hz, 20 s -> center-cropped to the middle 10 s.
    pre = make_preprocessor()
    signal = np.random.randn(12, 2000).astype(np.float32)
    out = pre(signal, sampling_rate=100)
    assert out.shape == (12, 1000)
    np.testing.assert_allclose(out, signal[:, 500:1500], rtol=0, atol=1e-6)


def test_short_signal_is_zero_padded():
    # 100 Hz, 6 s -> centered in a 10 s window with zero padding either side.
    pre = make_preprocessor()
    signal = np.random.randn(12, 600).astype(np.float32)
    out = pre(signal, sampling_rate=100)
    assert out.shape == (12, 1000)
    np.testing.assert_allclose(out[:, :200], 0.0, atol=0)
    np.testing.assert_allclose(out[:, 800:], 0.0, atol=0)
    np.testing.assert_allclose(out[:, 200:800], signal, atol=1e-6)


def test_standardization_uses_training_stats():
    # With per-lead mean/std fit on the input, the output is zero-mean, unit-std.
    signal = np.random.randn(12, 1000).astype(np.float32) * 3.0 + 5.0
    mean = signal.mean(axis=1)
    std = signal.std(axis=1)
    out = make_preprocessor(mean, std)(signal, sampling_rate=100)
    np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-4)
    np.testing.assert_allclose(out.std(axis=1), 1.0, atol=1e-3)


@pytest.mark.parametrize("n_leads", [11, 13])
def test_wrong_lead_count_rejected(n_leads):
    with pytest.raises(ValueError):
        make_preprocessor()(np.zeros((n_leads, 1000), dtype=np.float32), sampling_rate=100)


def test_non_finite_rejected():
    signal = np.zeros((12, 1000), dtype=np.float32)
    signal[0, 0] = np.nan
    with pytest.raises(ValueError):
        make_preprocessor()(signal, sampling_rate=100)


def test_empty_signal_rejected():
    with pytest.raises(ValueError):
        make_preprocessor()(np.zeros((12, 0), dtype=np.float32), sampling_rate=100)
