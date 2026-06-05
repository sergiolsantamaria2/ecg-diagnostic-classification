"""Self-supervised pretraining loop with best-checkpoint by validation loss."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader


class PretrainTrainer:
    """Mixed-precision trainer for self-supervised pretext tasks.

    Mirrors the supervised trainer (bf16 autocast, cosine schedule, W&B logging)
    but selects the checkpoint with the lowest validation loss and saves the
    encoder weights separately — that encoder is the asset reused downstream for
    fine-tuning and linear probing.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_step: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        device: str | None = None,
        amp: bool = True,
        ckpt_dir: str | Path = "checkpoints",
        logger: Callable[[dict], None] | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_step = loss_step
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.amp = amp and self.device == "cuda"
        self.ckpt_dir = Path(ckpt_dir)
        self.logger = logger
        self.best_val_loss = float("inf")

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        running, n = 0.0, 0
        for x, _ in loader:
            x = x.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                loss = self.loss_step(self.model, x)
            loss.backward()
            self.optimizer.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        return running / n

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        running, n = 0.0, 0
        for x, _ in loader:
            x = x.to(self.device, non_blocking=True)
            with self._autocast():
                loss = self.loss_step(self.model, x)
            running += loss.item() * x.size(0)
            n += x.size(0)
        return running / n

    def save_checkpoint(self, path: Path, **extra) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "encoder_state": self.model.encoder.state_dict(),
                **extra,
            },
            path,
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> float:
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            if self.scheduler is not None:
                self.scheduler.step()

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.save_checkpoint(self.ckpt_dir / "best.pt", epoch=epoch, val_loss=val_loss)

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            if self.logger is not None:
                self.logger(record)
            print(
                f"epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                f"val_loss {val_loss:.4f}" + ("  *best" if is_best else "")
            )
        return self.best_val_loss
