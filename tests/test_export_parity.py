"""The exported ONNX graph must match PyTorch within tolerance, for every encoder."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest
import torch

from deep_ecg.data.labels import SUPERCLASSES
from deep_ecg.models import build_model
from deep_ecg.serving.export import export_onnx, verify_parity

from .test_models import ENCODERS

ATOL, RTOL = 1e-4, 1e-3


def _export(tmp_path, name="simple", served_len=1000):
    model = build_model(
        {"name": name, "args": ENCODERS[name]}, {"name": "linear"}, num_classes=len(SUPERCLASSES)
    )
    onnx_path = tmp_path / "model.onnx"
    export_onnx(model, served_len=served_len, onnx_path=onnx_path)
    return model, onnx_path


@pytest.mark.parametrize("name", ENCODERS)
@pytest.mark.parametrize("served_len", [1000, 500])  # full window and a training crop
def test_onnx_matches_pytorch(tmp_path, name, served_len):
    model, onnx_path = _export(tmp_path, name, served_len)
    # verify_parity raises if any input is out of tolerance (it checks two batch sizes).
    assert verify_parity(model, onnx_path, served_len, atol=ATOL, rtol=RTOL) < ATOL


def test_dynamic_batch_axis(tmp_path):
    model, onnx_path = _export(tmp_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(3, 12, 1000)  # a batch size the graph was not traced with
    with torch.no_grad():
        ref = model(x).numpy()
    onnx_out = session.run(None, {"signal": x.numpy()})[0]
    assert onnx_out.shape == (3, len(SUPERCLASSES))
    np.testing.assert_allclose(onnx_out, ref, atol=ATOL, rtol=RTOL)
