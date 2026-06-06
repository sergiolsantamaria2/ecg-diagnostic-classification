"""Config-driven self-supervised pretraining entrypoint (Hydra).

Pretrains the shared encoder on unlabeled ECG with a masked or contrastive pretext
task and saves the encoder weights for downstream fine-tuning and linear probing.
Uses the official train folds (1-8) as the pretraining corpus; an online linear
probe on the validation fold (9) selects the checkpoint and early-stops, since the
pretext loss is an imperfect proxy for downstream quality. Fold 10 is never used.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from deep_ecg.data.dataset import build_contrastive_datasets, build_datasets
from deep_ecg.data.sources import build_source
from deep_ecg.models.encoders import build_encoder
from deep_ecg.ssl import build_loss_step, build_pretext
from deep_ecg.training.pretrainer import PretrainTrainer
from deep_ecg.training.probe import linear_probe_auroc
from deep_ecg.training.schedulers import build_scheduler
from deep_ecg.utils.seed import seed_everything, seed_worker


@hydra.main(version_base=None, config_path="configs", config_name="pretrain")
def main(cfg: DictConfig) -> float:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    run_dir = Path(HydraConfig.get().run.dir)

    # -- data (no labels used; unlabeled records kept) ---------------------
    source = build_source(
        cfg.data.source,
        root=cfg.data.root,
        sampling_rate=cfg.data.sampling_rate,
        drop_unlabeled=cfg.data.drop_unlabeled,
    )
    ssl_cfg = OmegaConf.to_container(cfg.ssl, resolve=True)
    name = ssl_cfg["name"]
    if name == "contrastive":
        views = ssl_cfg["views"]
        train_ds, val_ds, stats = build_contrastive_datasets(
            source, crop_len=views["crop_len"], augmentations=views["augmentations"]
        )
    else:
        augmentations = OmegaConf.to_container(cfg.data.augmentations, resolve=True)
        train_ds, val_ds, _, stats = build_datasets(source, augmentations=augmentations)
    np.savez(run_dir / "lead_stats.npz", **stats)

    def loader(ds, shuffle, drop_last=False):
        kwargs = {}
        if shuffle:
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

    # -- downstream probe monitor (labeled folds, no augmentation) ----------
    probe_fn = None
    monitor = cfg.trainer.monitor
    if monitor.enabled:
        labeled = build_source(
            cfg.data.source,
            root=cfg.data.root,
            sampling_rate=cfg.data.sampling_rate,
            drop_unlabeled=True,
        )
        probe_train, probe_val, _, _ = build_datasets(labeled, augmentations=None)
        probe_train_loader = loader(probe_train, shuffle=False)
        probe_val_loader = loader(probe_val, shuffle=False)

        def probe_fn(encoder):
            return linear_probe_auroc(
                encoder,
                probe_train_loader,
                probe_val_loader,
                labeled.classes,
                cfg.device,
                epochs=monitor.probe_epochs,
                lr=monitor.probe_lr,
            )

    # -- encoder + pretext model -------------------------------------------
    encoder_cfg = OmegaConf.to_container(cfg.model.encoder, resolve=True)
    encoder = build_encoder(encoder_cfg["name"], **(encoder_cfg.get("args") or {}))
    model = build_pretext(name, encoder, ssl_cfg)
    loss_step = build_loss_step(name, ssl_cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay
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

    # -- pretrain ----------------------------------------------------------
    trainer = PretrainTrainer(
        model,
        loss_step,
        optimizer,
        scheduler=scheduler,
        device=cfg.device,
        amp=cfg.trainer.amp,
        ckpt_dir=run_dir / "checkpoints",
        logger=logger,
        probe=probe_fn,
        eval_every=monitor.eval_every,
        patience=monitor.patience,
    )
    best_val = trainer.fit(train_loader, val_loader, epochs=cfg.trainer.epochs)
    print(f"\nbest val {name} loss: {best_val:.4f}")
    if probe_fn is not None:
        print(f"best probe macro-AUROC: {trainer.best_probe_auroc:.4f}")

    if logger is not None:
        logger.summary({"best_val_loss": best_val, "best_probe_auroc": trainer.best_probe_auroc})
        logger.finish()

    return best_val


if __name__ == "__main__":
    main()
