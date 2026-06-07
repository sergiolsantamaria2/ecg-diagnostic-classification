"""Evaluate a trained run on the test fold with per-class error analysis.

Loads a run's Hydra config and best checkpoint, rebuilds the data and model,
tunes per-class thresholds on the validation fold, and reports test metrics and
a per-class report. Usage: ``uv run python scripts/eval/evaluate.py <run_dir>``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from deep_ecg.data.dataset import build_datasets
from deep_ecg.data.sources import build_source
from deep_ecg.evaluation.analysis import per_class_report, tune_thresholds
from deep_ecg.evaluation.metrics import compute_metrics
from deep_ecg.models import build_model
from deep_ecg.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="training output directory")
    parser.add_argument(
        "--tta-crop-len",
        type=int,
        default=None,
        help="evaluate with test-time crop averaging of this length",
    )
    parser.add_argument("--tta-n-crops", type=int, default=5)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run_dir / ".hydra" / "config.yaml")
    source = build_source(
        cfg.data.source,
        root=cfg.data.root,
        sampling_rate=cfg.data.sampling_rate,
        drop_unlabeled=cfg.data.drop_unlabeled,
    )
    _, val_ds, test_ds, _ = build_datasets(source, augmentations=None)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = build_model(
        encoder=model_cfg["encoder"],
        head=model_cfg["head"],
        num_classes=cfg.task.num_classes,
    )
    ckpt = torch.load(args.run_dir / "checkpoints" / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    trainer = Trainer(model, class_names=source.classes)

    def predict(ds):
        loader = DataLoader(ds, batch_size=256, num_workers=8, pin_memory=True)
        if args.tta_crop_len:
            return trainer.predict_tta(loader, args.tta_crop_len, args.tta_n_crops)
        return trainer.predict(loader)

    y_val, s_val = predict(val_ds)
    y_test, s_test = predict(test_ds)

    metrics = compute_metrics(y_test, s_test, source.classes)
    thresholds = tune_thresholds(y_val, s_val, source.classes)
    report = per_class_report(y_test, s_test, source.classes, thresholds)

    print(f"\nrun: {args.run_dir}  (best epoch {ckpt.get('epoch', '?')})")
    print(f"test macro-AUROC: {metrics['macro_auroc']:.4f}")
    print("per-class AUROC :", {k: round(v, 4) for k, v in metrics["per_class_auroc"].items()})
    print("\nper-class report (thresholds tuned on validation):")
    header = f"{'class':6s} {'thr':>5s} {'prec':>7s} {'recall':>7s} {'f1':>7s} {'support':>8s}"
    print(header)
    for r in report:
        print(
            f"{r['class']:6s} {r['threshold']:5.2f} {r['precision']:7.4f} "
            f"{r['recall']:7.4f} {r['f1']:7.4f} {r['support']:8d}"
        )

    out = {
        "run_dir": str(args.run_dir),
        "best_epoch": ckpt.get("epoch"),
        "macro_auroc": metrics["macro_auroc"],
        "per_class_auroc": metrics["per_class_auroc"],
        "thresholds": thresholds,
        "per_class_report": report,
    }
    (args.run_dir / "evaluation.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {args.run_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
