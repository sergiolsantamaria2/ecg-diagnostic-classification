"""Shared test fixtures: a tiny serving bundle built from a random model.

The bundle mirrors a real one (ONNX + TorchScript + metadata + lead statistics)
but uses a small randomly initialized model, so the serving tests never depend on
a trained checkpoint or on PTB-XL being present.
"""

from __future__ import annotations

import numpy as np
import pytest

from deep_ecg.data.labels import SUPERCLASSES
from deep_ecg.models import build_model
from deep_ecg.serving.bundle import BundleMetadata, ModelSpec, ServingBundle
from deep_ecg.serving.export import export_model_files

TARGET_LEN = 1000


@pytest.fixture
def tiny_bundle(tmp_path):
    """A complete serving bundle in a temporary directory; returns its path."""
    model = build_model({"name": "simple"}, {"name": "linear"}, num_classes=len(SUPERCLASSES))
    out = tmp_path / "bundle"
    out.mkdir()
    export_model_files(model, TARGET_LEN, out / "model_0.onnx", out / "model_0.ts")
    metadata = BundleMetadata(
        name="test-model",
        classes=list(SUPERCLASSES),
        thresholds={c: 0.5 for c in SUPERCLASSES},
        sampling_rate=100,
        target_len=TARGET_LEN,
        models=[ModelSpec(onnx="model_0.onnx", torchscript="model_0.ts")],
    )
    bundle = ServingBundle(
        out, metadata, np.zeros(12, dtype=np.float32), np.ones(12, dtype=np.float32)
    )
    bundle.save()
    return out
