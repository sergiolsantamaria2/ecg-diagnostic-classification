"""Config-driven supervised training entrypoint (Hydra)."""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from deep_ecg.data.dataset import build_datasets
from deep_ecg.data.sources import build_source
from deep_ecg.models import build_model, freeze_encoder
from deep_ecg.training.losses import build_loss
from deep_ecg.training.schedulers import build_scheduler
from deep_ecg.training.trainer import Trainer
from deep_ecg.utils.seed import seed_everything, seed_worker


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    run_dir = Path(HydraConfig.get().run.dir)

    # -- data --------------------------------------------------------------
    source = build_source(
        cfg.data.source,
        root=cfg.data.root,
        sampling_rate=cfg.data.sampling_rate,
        drop_unlabeled=cfg.data.drop_unlabeled,
    )
    augmentations = OmegaConf.to_container(cfg.data.augmentations, resolve=True)
    train_ds, val_ds, test_ds, stats = build_datasets(
        source,
        augmentations=augmentations,
        label_fraction=cfg.data.label_fraction,
        subsample_seed=cfg.data.subsample_seed,
    )
    print(f"labeled train size: {len(train_ds)} (fraction {cfg.data.label_fraction})")
    np.savez(run_dir / "lead_stats.npz", **stats)

    def loader(ds, shuffle, drop_last=False):
        kwargs = {}
        if shuffle:  # decorrelated, reproducible augmentation + shuffling
            kwargs["worker_init_fn"] = seed_worker
            kwargs["generator"] = torch.Generator().manual_seed(cfg.seed)
        return DataLoader(
            ds,
            batch_size=cfg.data.batch_size,
            shuffle=shuffle,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            drop_last=drop_last,
            **kwargs,
        )

    train_loader = loader(train_ds, shuffle=True, drop_last=True)
    val_loader = loader(val_ds, shuffle=False)
    test_loader = loader(test_ds, shuffle=False)

    # -- model / loss / optimizer / scheduler ------------------------------
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = build_model(
        encoder=model_cfg["encoder"],
        head=model_cfg["head"],
        num_classes=cfg.task.num_classes,
    )

    # Optionally start from a self-supervised encoder and freeze it (linear probe).
    if cfg.model.pretrained_ckpt:
        ckpt = torch.load(cfg.model.pretrained_ckpt, map_location="cpu")
        model.encoder.load_state_dict(ckpt["encoder_state"])
        print(f"loaded pretrained encoder from {cfg.model.pretrained_ckpt}")
    if cfg.model.freeze_encoder:
        freeze_encoder(model)
        print("encoder frozen (linear probe)")

    loss_fn = build_loss(cfg.task.loss.name)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay
    )
    sched_cfg = OmegaConf.to_container(cfg.trainer.scheduler, resolve=True)
    scheduler = build_scheduler(sched_cfg["name"], optimizer, **(sched_cfg.get("args") or {}))

    # -- logging -----------------------------------------------------------
    logger = None
    if cfg.wandb.enabled:
        from deep_ecg.utils.logging import WandbLogger

        logger = WandbLogger(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            group=cfg.wandb.group,
            name=cfg.wandb.name,
            tags=list(cfg.wandb.tags),
            notes=cfg.wandb.notes,
        )

    # -- train -------------------------------------------------------------
    trainer = Trainer(
        model,
        loss_fn,
        optimizer,
        class_names=source.classes,
        scheduler=scheduler,
        device=cfg.device,
        amp=cfg.trainer.amp,
        ckpt_dir=run_dir / "checkpoints",
        logger=logger,
    )
    best_val = trainer.fit(train_loader, val_loader, epochs=cfg.trainer.epochs)

    # -- test with the best checkpoint -------------------------------------
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=cfg.device)
    trainer.model.load_state_dict(ckpt["model_state"])
    test_metrics = trainer.evaluate(test_loader)
    print(f"\nbest val macro-AUROC: {best_val:.4f}")
    print(f"test macro-AUROC    : {test_metrics['macro_auroc']:.4f}")
    print(
        "test per-class AUROC :",
        {k: round(v, 4) for k, v in test_metrics["per_class_auroc"].items()},
    )

    if logger is not None:
        logger.summary(
            {
                "best_val_macro_auroc": best_val,
                "test_macro_auroc": test_metrics["macro_auroc"],
                "test_macro_f1": test_metrics["macro_f1"],
                **{f"test_auroc/{k}": v for k, v in test_metrics["per_class_auroc"].items()},
            }
        )
        logger.finish()

    return test_metrics["macro_auroc"]


if __name__ == "__main__":
    main()
