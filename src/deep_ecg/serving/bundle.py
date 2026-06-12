"""Self-describing serving bundle: everything needed to run inference.

A bundle is a directory holding the exported model file(s), the training lead
statistics, and a ``metadata.json`` describing the classes, decision thresholds,
expected input format and per-model inference settings. It is produced by the
export script and consumed by the predictor and the API, so serving depends only
on the bundle — never on the Hydra run directory, the checkpoint or W&B.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..data.sources.base import CANONICAL_LEADS

FORMAT_VERSION = 1
METADATA_FILE = "metadata.json"
STATS_FILE = "lead_stats.npz"


@dataclass
class ModelSpec:
    """One exported model and how to run it.

    ``crop_len`` (in samples at the bundle's sampling rate) marks a model trained
    with random cropping: it is served with test-time crop averaging over
    ``n_crops`` evenly spaced windows, matching the evaluation protocol. A null
    ``crop_len`` means the model runs once on the full-length signal.
    """

    onnx: str
    torchscript: str
    crop_len: int | None = None
    n_crops: int = 1
    source_run: str | None = None


@dataclass
class BundleMetadata:
    """Static description of a serving bundle (serialized to ``metadata.json``)."""

    name: str
    classes: list[str]
    thresholds: dict[str, float]
    sampling_rate: int
    target_len: int
    models: list[ModelSpec]
    lead_order: list[str] = field(default_factory=lambda: list(CANONICAL_LEADS))
    format_version: int = FORMAT_VERSION
    parity: dict | None = None


class ServingBundle:
    """A loaded bundle: metadata plus the lead statistics, rooted at a directory."""

    def __init__(self, root: Path, metadata: BundleMetadata, mean: np.ndarray, std: np.ndarray):
        self.root = Path(root)
        self.metadata = metadata
        self.mean = mean
        self.std = std

    # -- paths --------------------------------------------------------------
    def onnx_path(self, spec: ModelSpec) -> Path:
        return self.root / spec.onnx

    def torchscript_path(self, spec: ModelSpec) -> Path:
        return self.root / spec.torchscript

    # -- io -----------------------------------------------------------------
    @classmethod
    def load(cls, root: str | Path) -> ServingBundle:
        root = Path(root)
        raw = json.loads((root / METADATA_FILE).read_text())
        raw["models"] = [ModelSpec(**m) for m in raw["models"]]
        metadata = BundleMetadata(**raw)
        if metadata.format_version != FORMAT_VERSION:
            raise ValueError(
                f"unsupported bundle format {metadata.format_version} "
                f"(this build expects {FORMAT_VERSION})"
            )
        stats = np.load(root / STATS_FILE)
        return cls(root, metadata, stats["mean"], stats["std"])

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = asdict(self.metadata)
        (self.root / METADATA_FILE).write_text(json.dumps(payload, indent=2))
        np.savez(self.root / STATS_FILE, mean=self.mean, std=self.std)
