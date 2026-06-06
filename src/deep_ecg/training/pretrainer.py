"""Self-supervised pretraining loop with best-checkpoint by validation loss."""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
        loss_step: Callable[[torch.nn.Module, Sequence[torch.Tensor]], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        device: str | None = None,
        amp: bool = True,
        ckpt_dir: str | Path = "checkpoints",
        logger: Callable[[dict], None] | None = None,
        probe: Callable[[torch.nn.Module], float] | None = None,
        eval_every: int = 10,
        patience: int = 3,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_step = loss_step
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.amp = amp and self.device == "cuda"
        self.ckpt_dir = Path(ckpt_dir)
        self.logger = logger
        self.probe = probe
        self.eval_every = eval_every
        self.patience = patience
        self.best_val_loss = float("inf")
        self.best_probe_auroc = -float("inf")

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp)

    def _to_device(self, batch: Sequence) -> list:
        return [b.to(self.device, non_blocking=True) if torch.is_tensor(b) else b for b in batch]

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        running, n = 0.0, 0
        for batch in loader:
            batch = self._to_device(batch)
            bs = batch[0].size(0)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                loss = self.loss_step(self.model, batch)
            loss.backward()
            self.optimizer.step()
            running += loss.item() * bs
            n += bs
        return running / n

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        running, n = 0.0, 0
        for batch in loader:
            batch = self._to_device(batch)
            bs = batch[0].size(0)
            with self._autocast():
                loss = self.loss_step(self.model, batch)
            running += loss.item() * bs
            n += bs
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
        """Pretrain for up to ``epochs``.

        The checkpoint with the lowest pretext loss is saved as ``best_pretext``.
        When a downstream probe is configured, its validation macro-AUROC is
        evaluated every ``eval_every`` epochs; the best is saved as ``best_probe``
        and pretraining stops once it has not improved for ``patience`` probes.
        """
        no_improve = 0
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            if self.scheduler is not None:
                self.scheduler.step()

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    self.ckpt_dir / "best_pretext.pt", epoch=epoch, val_loss=val_loss
                )

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            }

            probe_auroc = None
            if self.probe is not None and epoch % self.eval_every == 0:
                probe_auroc = self.probe(self.model.encoder)
                record["probe_auroc"] = probe_auroc
                if probe_auroc > self.best_probe_auroc:
                    self.best_probe_auroc = probe_auroc
                    self.save_checkpoint(
                        self.ckpt_dir / "best_probe.pt", epoch=epoch, probe_auroc=probe_auroc
                    )
                    no_improve = 0
                else:
                    no_improve += 1

            if self.logger is not None:
                self.logger(record)
            line = f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}"
            if is_best:
                line += "  *best"
            if probe_auroc is not None:
                line += f" | probe-AUROC {probe_auroc:.4f}"
                if no_improve == 0:
                    line += "  *best-probe"
            print(line)

            if self.probe is not None and no_improve >= self.patience:
                print(f"early stop: probe-AUROC flat for {self.patience} evaluations")
                break
        return self.best_val_loss
