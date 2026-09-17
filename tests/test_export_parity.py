"""The exported ONNX graph must match PyTorch within tolerance."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import torch

from deep_ecg.data.labels import SUPERCLASSES
from deep_ecg.models import build_model
from deep_ecg.serving.export import export_onnx, verify_parity

ATOL, RTOL = 1e-4, 1e-3


def _export(tmp_path):
    model = build_model({"name": "simple"}, {"name": "linear"}, num_classes=len(SUPERCLASSES))
    onnx_path = tmp_path / "model.onnx"
    export_onnx(model, target_len=1000, onnx_path=onnx_path)
    return model, onnx_path


def test_onnx_matches_pytorch(tmp_path):
    model, onnx_path = _export(tmp_path)
    # verify_parity raises if any input is out of tolerance (it checks two shapes).
    assert verify_parity(model, onnx_path, target_len=1000, atol=ATOL, rtol=RTOL) < ATOL


def test_dynamic_axes_allow_other_lengths(tmp_path):
    model, onnx_path = _export(tmp_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(3, 12, 1300)  # different batch and length than the trace
    with torch.no_grad():
        ref = model(x).numpy()
    onnx_out = session.run(None, {"signal": x.numpy()})[0]
    assert onnx_out.shape == (3, len(SUPERCLASSES))
    np.testing.assert_allclose(onnx_out, ref, atol=ATOL, rtol=RTOL)
