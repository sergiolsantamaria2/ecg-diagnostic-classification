"""Export trained run(s) to ONNX as a serving bundle.

Loads each run's Hydra config and best checkpoint, rebuilds the model, exports it
to ONNX (dynamic batch axis, traced at the length it is served at) and verifies
that the graph matches
PyTorch on random inputs within tolerance. The lead statistics, class names,
decision thresholds and per-model inference settings are written alongside as a
:class:`~deep_ecg.serving.bundle.ServingBundle`.

Single model::

    uv run python scripts/export_model.py outputs/<run> --out artifacts/resnet1d

Full ensemble (faithful to the results table)::

    uv run python scripts/export_model.py outputs/<run1> outputs/<run2> ... \
        --out artifacts/ensemble --thresholds outputs/ensemble.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from deep_ecg.data.labels import SUPERCLASSES
from deep_ecg.models import build_model
from deep_ecg.serving.bundle import BundleMetadata, ModelSpec, ServingBundle
from deep_ecg.serving.export import export_onnx, verify_parity


def load_run_model(run_dir: Path) -> tuple[torch.nn.Module, dict]:
    """Rebuild a run's model with its best weights, in eval mode on CPU."""
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = build_model(
        encoder=model_cfg["encoder"],
        head=model_cfg["head"],
        num_classes=cfg.task.num_classes,
    )
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return model.float().eval(), cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="training runs to export")
    parser.add_argument("--out", type=Path, required=True, help="bundle output directory")
    parser.add_argument("--name", type=str, default=None, help="bundle name (default: out dir)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--n-crops", type=int, default=5, help="TTA crops for crop-trained models")
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="JSON with a 'thresholds' map (e.g. an ensemble.json); "
        "defaults to the first run's evaluation.json, else 0.5",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    specs: list[ModelSpec] = []
    means, stds, parities = [], [], []
    classes = sampling_rate = target_len = None
    for i, run_dir in enumerate(args.run_dirs):
        model, cfg = load_run_model(run_dir)
        stats = np.load(run_dir / "lead_stats.npz")
        means.append(stats["mean"])
        stds.append(stats["std"])

        run_sr = int(cfg.data.sampling_rate)
        if cfg.task.num_classes != len(SUPERCLASSES):
            raise ValueError(f"unexpected num_classes {cfg.task.num_classes}")
        # PTB-XL records are a fixed 10 s; the canonical serving length follows.
        run_target_len = run_sr * 10
        if classes is None:
            classes, sampling_rate, target_len = list(SUPERCLASSES), run_sr, run_target_len
        elif (run_sr, run_target_len) != (sampling_rate, target_len):
            raise ValueError("runs disagree on sampling rate / length; cannot bundle together")

        # A crop-trained model is served on crops, so its graph is traced at that length.
        crop = cfg.data.augmentations.get("crop")
        crop_len = int(crop) if crop else None
        served_len = crop_len or run_target_len
        onnx_path = args.out / f"model_{i}.onnx"
        export_onnx(model, served_len, onnx_path, args.opset)
        parity = verify_parity(model, onnx_path, served_len, atol=args.atol, rtol=args.rtol)
        parities.append(parity)

        specs.append(
            ModelSpec(
                onnx=onnx_path.name,
                crop_len=crop_len,
                n_crops=args.n_crops if crop_len else 1,
                source_run=str(run_dir),
            )
        )
        print(f"[{i}] {run_dir.name}: exported (crop={crop_len}) max|Δ| onnx={parity:.2e}")

    # Standardization stats must agree across runs so one preprocessing pass feeds
    # every model in the ensemble; the baselines all fit them on the full train folds.
    mean, std = means[0], stds[0]
    for m, s in zip(means[1:], stds[1:], strict=True):
        if not (np.allclose(m, mean) and np.allclose(s, std)):
            raise ValueError("runs have different lead statistics; cannot share preprocessing")

    thresholds = _resolve_thresholds(args.thresholds, args.run_dirs[0], classes)

    metadata = BundleMetadata(
        name=args.name or args.out.name,
        classes=classes,
        thresholds=thresholds,
        sampling_rate=sampling_rate,
        target_len=target_len,
        models=specs,
        parity={"atol": args.atol, "rtol": args.rtol, "max_abs_diff_per_model": parities},
    )
    ServingBundle(args.out, metadata, mean, std).save()
    print(f"\nsaved bundle ({len(specs)} model(s)) to {args.out}")


def _resolve_thresholds(path: Path | None, first_run: Path, classes: list[str]) -> dict[str, float]:
    """Decision thresholds from an explicit file, the first run's eval, or 0.5.

    Both ``evaluate.py`` and ``ensemble.py`` write a ``thresholds`` map; older
    ensemble files only carry them inside ``per_class_report``.
    """
    source = path if path is not None else first_run / "evaluation.json"
    if source.exists():
        data = json.loads(source.read_text())
        thresholds = data.get("thresholds") or {
            row["class"]: row["threshold"] for row in data.get("per_class_report", [])
        }
        if thresholds:
            return {c: float(thresholds[c]) for c in classes}
    print("no thresholds found; defaulting to 0.5 per class")
    return {c: 0.5 for c in classes}


if __name__ == "__main__":
    main()
