"""ONNX and TorchScript must match PyTorch within tolerance."""

from __future__ import annotations

import numpy as np

from deep_ecg.data.labels import SUPERCLASSES
from deep_ecg.models import build_model
from deep_ecg.serving.export import export_model_files, verify_parity

ATOL, RTOL = 1e-4, 1e-3


def test_onnx_and_torchscript_match_pytorch(tmp_path):
    model = build_model({"name": "simple"}, {"name": "linear"}, num_classes=len(SUPERCLASSES))
    onnx_path = tmp_path / "model.onnx"
    ts_path = tmp_path / "model.ts"
    export_model_files(model, target_len=1000, onnx_path=onnx_path, ts_path=ts_path)

    # verify_parity raises if any input is out of tolerance (it checks two shapes).
    diffs = verify_parity(model, onnx_path, ts_path, target_len=1000, atol=ATOL, rtol=RTOL)
    assert diffs["max_abs_diff_onnx"] < ATOL
    assert diffs["max_abs_diff_torchscript"] < ATOL


def test_dynamic_axes_allow_other_lengths(tmp_path):
    import onnxruntime as ort
    import torch

    model = build_model({"name": "simple"}, {"name": "linear"}, num_classes=len(SUPERCLASSES))
    onnx_path = tmp_path / "model.onnx"
    ts_path = tmp_path / "model.ts"
    export_model_files(model, target_len=1000, onnx_path=onnx_path, ts_path=ts_path)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(3, 12, 1300)  # different batch and length than the trace
    with torch.no_grad():
        ref = model(x).numpy()
    onnx_out = session.run(None, {"signal": x.numpy()})[0]
    assert onnx_out.shape == (3, len(SUPERCLASSES))
    np.testing.assert_allclose(onnx_out, ref, atol=ATOL, rtol=RTOL)
