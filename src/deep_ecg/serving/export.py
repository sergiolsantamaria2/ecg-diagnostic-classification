"""Export a PyTorch model to ONNX and check numerical parity against PyTorch.

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


def export_onnx(model: torch.nn.Module, target_len: int, onnx_path: Path, opset: int = 17) -> None:
    """Write the ONNX graph for ``model`` in eval mode."""
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


def verify_parity(
    model: torch.nn.Module,
    onnx_path: Path,
    target_len: int,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> float:
    """Compare ONNX Runtime output to PyTorch; raise if out of tolerance.

    Two inputs are checked — the nominal ``(1, 12, target_len)`` and a larger
    ``(2, 12, target_len + 300)`` batch — so the dynamic batch and time axes are
    exercised, not just the shape the model was traced with. Returns the maximum
    absolute deviation seen.
    """
    model = model.eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_diff = 0.0
    for shape in [(1, N_LEADS, target_len), (2, N_LEADS, target_len + 300)]:
        x = torch.randn(*shape)
        with torch.no_grad():
            ref = model(x).numpy()
        out = session.run(None, {"signal": x.numpy()})[0]
        max_diff = max(max_diff, float(np.abs(out - ref).max()))
        np.testing.assert_allclose(out, ref, atol=atol, rtol=rtol)
    return max_diff
