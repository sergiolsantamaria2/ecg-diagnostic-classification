"""Export a PyTorch model to ONNX and check numerical parity against PyTorch.

Kept separate from the CLI so the export and the parity check are importable and
unit-tested. The batch axis is dynamic; the time axis is fixed at the length the
model is served at (the full window, or the training crop for crop-trained
models). The legacy exporter bakes the sequence length into the Transformer's
attention reshape, so a graph traced at one length is only valid at that length.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

N_LEADS = 12


def export_onnx(model: torch.nn.Module, served_len: int, onnx_path: Path, opset: int = 17) -> None:
    """Write the ONNX graph for ``model`` in eval mode, traced at ``served_len``."""
    model = model.eval()
    dummy = torch.randn(1, N_LEADS, served_len)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["signal"],
        output_names=["logits"],
        dynamic_axes={"signal": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )


def verify_parity(
    model: torch.nn.Module,
    onnx_path: Path,
    served_len: int,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> float:
    """Compare ONNX Runtime output to PyTorch; raise if out of tolerance.

    Checks a single sample and a batch of two at ``served_len`` so the dynamic
    batch axis is exercised, not just the shape the model was traced with.
    Returns the maximum absolute deviation seen.
    """
    model = model.eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_diff = 0.0
    for batch in (1, 2):
        x = torch.randn(batch, N_LEADS, served_len)
        with torch.no_grad():
            ref = model(x).numpy()
        out = session.run(None, {"signal": x.numpy()})[0]
        max_diff = max(max_diff, float(np.abs(out - ref).max()))
        np.testing.assert_allclose(out, ref, atol=atol, rtol=rtol)
    return max_diff
