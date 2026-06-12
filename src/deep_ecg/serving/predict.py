"""Inference over a serving bundle.

Wraps a :class:`~deep_ecg.serving.bundle.ServingBundle` with the preprocessing and
the runtime. Inference prefers ONNX Runtime and falls back to the TorchScript
model under PyTorch if ONNX Runtime is unavailable. A crop-trained model is served
with test-time crop averaging; several models are ensembled by averaging their
probabilities, matching the offline evaluation protocol.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .bundle import ModelSpec, ServingBundle
from .preprocess import Preprocessor


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class _OnnxRunner:
    """Run one exported model with ONNX Runtime (CPU)."""

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x.astype(np.float32)})[0]


class _TorchScriptRunner:
    """Run one exported model with the TorchScript interpreter (fallback)."""

    def __init__(self, path: Path) -> None:
        import torch

        self._torch = torch
        self.model = torch.jit.load(str(path)).eval()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            return self.model(self._torch.from_numpy(x.astype(np.float32))).numpy()


def _make_runners(bundle: ServingBundle, backend: str):
    """Build a runner per model; resolve ``auto`` to ONNX with TorchScript fallback."""
    specs = bundle.metadata.models
    if backend in ("auto", "onnx"):
        try:
            runners = [_OnnxRunner(bundle.onnx_path(s)) for s in specs]
            return runners, "onnx"
        except Exception:
            if backend == "onnx":
                raise
    runners = [_TorchScriptRunner(bundle.torchscript_path(s)) for s in specs]
    return runners, "torchscript"


class Predictor:
    """Turn a raw ``(12, L)`` ECG into per-class probabilities and decisions."""

    def __init__(self, bundle: ServingBundle, backend: str = "auto") -> None:
        self.bundle = bundle
        self.meta = bundle.metadata
        self.preprocessor = Preprocessor(
            bundle.mean,
            bundle.std,
            target_fs=self.meta.sampling_rate,
            target_len=self.meta.target_len,
        )
        self.runners, self.backend = _make_runners(bundle, backend)

    @classmethod
    def from_dir(cls, root: str | Path, backend: str = "auto") -> Predictor:
        return cls(ServingBundle.load(root), backend=backend)

    def predict(self, signal: np.ndarray, sampling_rate: int) -> dict:
        """Probabilities, multi-label decision and the thresholds used."""
        x = self.preprocessor(signal, sampling_rate)
        probs = self._probabilities(x)
        thresholds = self.meta.thresholds
        classes = self.meta.classes
        return {
            "probabilities": {c: float(probs[i]) for i, c in enumerate(classes)},
            "labels": {c: bool(probs[i] >= thresholds[c]) for i, c in enumerate(classes)},
            "thresholds": {c: float(thresholds[c]) for c in classes},
        }

    # -- internals ----------------------------------------------------------
    def _probabilities(self, x: np.ndarray) -> np.ndarray:
        """Ensemble-averaged probabilities over the bundle's models."""
        per_model = [
            self._model_probabilities(runner, x, spec)
            for runner, spec in zip(self.runners, self.meta.models, strict=True)
        ]
        return np.mean(per_model, axis=0)

    def _model_probabilities(self, runner, x: np.ndarray, spec: ModelSpec) -> np.ndarray:
        if spec.crop_len:
            return self._tta(runner, x, spec)
        return _sigmoid(runner(x[None]))[0]

    @staticmethod
    def _tta(runner, x: np.ndarray, spec: ModelSpec) -> np.ndarray:
        """Average the sigmoid over evenly spaced crops (test-time augmentation)."""
        length = x.shape[1]
        starts = np.linspace(0, length - spec.crop_len, spec.n_crops).astype(int)
        acc = None
        for start in starts:
            probs = _sigmoid(runner(x[None, :, start : start + spec.crop_len]))[0]
            acc = probs if acc is None else acc + probs
        return acc / len(starts)
