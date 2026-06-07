"""Ensemble several trained runs by averaging their test predictions.

Each run's best checkpoint is rebuilt from its Hydra config; runs trained with
random crop are evaluated with matching test-time crop averaging, others on the
full signal. Per-class thresholds are tuned on the (ensembled) validation fold.
Usage: ``uv run python scripts/eval/ensemble.py <run_dir> <run_dir> [...]``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from deep_ecg.data.dataset import build_datasets
from deep_ecg.data.sources import build_source
from deep_ecg.evaluation.analysis import per_class_report, tune_thresholds
from deep_ecg.evaluation.metrics import compute_metrics
from deep_ecg.models import build_model
from deep_ecg.training.trainer import Trainer


def run_predictions(run_dir: Path, val_loader, test_loader, classes, n_crops):
    """Validation and test probabilities for one run (auto TTA if crop-trained)."""
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = build_model(
        encoder=model_cfg["encoder"],
        head=model_cfg["head"],
        num_classes=cfg.task.num_classes,
    )
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    trainer = Trainer(model, class_names=classes)

    crop = cfg.data.augmentations.get("crop")
    if crop:

        def predict(loader):
            return trainer.predict_tta(loader, int(crop), n_crops)
    else:
        predict = trainer.predict
    return predict(val_loader), predict(test_loader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="run directories to ensemble")
    parser.add_argument("--n-crops", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("ensemble.json"))
    args = parser.parse_args()

    source = build_source("ptbxl", root="data/raw/ptbxl", sampling_rate=100, drop_unlabeled=True)
    _, val_ds, test_ds, _ = build_datasets(source, augmentations=None)

    def loader(ds):
        return DataLoader(ds, batch_size=256, num_workers=8, pin_memory=True)

    val_loader, test_loader = loader(val_ds), loader(test_ds)

    val_scores, test_scores = [], []
    y_val = y_test = None
    for run_dir in args.run_dirs:
        (y_val, s_val), (y_test, s_test) = run_predictions(
            run_dir, val_loader, test_loader, source.classes, args.n_crops
        )
        val_scores.append(s_val)
        test_scores.append(s_test)
        print(
            f"{run_dir.name}: test macro-AUROC "
            f"{compute_metrics(y_test, s_test, source.classes)['macro_auroc']:.4f}"
        )

    s_val_mean = np.mean(val_scores, axis=0)
    s_test_mean = np.mean(test_scores, axis=0)
    metrics = compute_metrics(y_test, s_test_mean, source.classes)
    thresholds = tune_thresholds(y_val, s_val_mean, source.classes)
    report = per_class_report(y_test, s_test_mean, source.classes, thresholds)

    print(f"\nensemble of {len(args.run_dirs)} runs")
    print(f"ensemble test macro-AUROC: {metrics['macro_auroc']:.4f}")
    print(
        "per-class AUROC          :",
        {k: round(v, 4) for k, v in metrics["per_class_auroc"].items()},
    )

    args.out.write_text(
        json.dumps(
            {
                "runs": [str(d) for d in args.run_dirs],
                "macro_auroc": metrics["macro_auroc"],
                "per_class_auroc": metrics["per_class_auroc"],
                "per_class_report": report,
            },
            indent=2,
        )
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
