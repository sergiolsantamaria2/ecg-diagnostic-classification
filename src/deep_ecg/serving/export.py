"""Export a PyTorch model to ONNX and TorchScript, and check numerical parity.

Kept separate from the CLI so the export and the parity check are importable and
unit-tested. ONNX is exported with dynamic batch and time axes so one graph serves
any batch size and signal length.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

N_LEADS = 12


def export_model_files(
    model: torch.nn.Module, target_len: int, onnx_path: Path, ts_path: Path, opset: int = 17
) -> None:
    """Write the ONNX and TorchScript files for ``model`` in eval mode."""
    model = model.eval()
    dummy = torch.randn(1, N_LEADS, target_len)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["signal"],
        output_names=["logits"],
        dynamic_axes={"signal": {0: "batch", 2: "time"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    torch.jit.save(torch.jit.trace(model, dummy), str(ts_path))


def verify_parity(
    model: torch.nn.Module,
    onnx_path: Path,
    ts_path: Path,
    target_len: int,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> dict:
    """Compare ONNX and TorchScript outputs to PyTorch; raise if out of tolerance.

    Two inputs are checked — the nominal ``(1, 12, target_len)`` and a larger
    ``(2, 12, target_len + 300)`` batch — so the dynamic batch and time axes are
    exercised, not just the shape the model was traced with. Returns the maximum
    absolute deviation seen for each backend.
    """
    model = model.eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ts_model = torch.jit.load(str(ts_path)).eval()
    max_onnx = max_ts = 0.0
    for shape in [(1, N_LEADS, target_len), (2, N_LEADS, target_len + 300)]:
        x = torch.randn(*shape)
        with torch.no_grad():
            ref = model(x).numpy()
            ts_out = ts_model(x).numpy()
        onnx_out = session.run(None, {"signal": x.numpy()})[0]
        max_onnx = max(max_onnx, float(np.abs(onnx_out - ref).max()))
        max_ts = max(max_ts, float(np.abs(ts_out - ref).max()))
        np.testing.assert_allclose(onnx_out, ref, atol=atol, rtol=rtol)
        np.testing.assert_allclose(ts_out, ref, atol=atol, rtol=rtol)
    return {"max_abs_diff_onnx": max_onnx, "max_abs_diff_torchscript": max_ts}
