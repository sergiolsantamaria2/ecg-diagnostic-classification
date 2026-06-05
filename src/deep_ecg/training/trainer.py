"""Supervised training loop with best-checkpoint selection by macro-AUROC."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..evaluation.metrics import compute_metrics


class Trainer:
    """Minimal mixed-precision trainer for multi-label classification.

    Trains with bf16 autocast (no gradient scaler needed on Ada), evaluates on
    the validation set each epoch, and keeps the checkpoint with the best
    validation macro-AUROC.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        class_names: Sequence[str] = (),
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        device: str | None = None,
        amp: bool = True,
        ckpt_dir: str | Path = "checkpoints",
        logger: Callable[[dict], None] | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device) if loss_fn is not None else None
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.class_names = list(class_names)
        self.amp = amp and self.device == "cuda"
        self.ckpt_dir = Path(ckpt_dir)
        self.logger = logger
        self.best_macro_auroc = -float("inf")

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        running, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                loss = self.loss_fn(self.model(x), y)
            loss.backward()
            self.optimizer.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        return running / n

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(y_true, y_score)`` with sigmoid probabilities."""
        self.model.eval()
        scores, targets = [], []
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            with self._autocast():
                logits = self.model(x)
            scores.append(torch.sigmoid(logits).float().cpu().numpy())
            targets.append(y.numpy())
        return np.concatenate(targets), np.concatenate(scores)

    @torch.no_grad()
    def predict_tta(
        self, loader: DataLoader, crop_len: int, n_crops: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Test-time augmentation: average sigmoid over evenly-spaced crops.

        Matches the cropped distribution the model was trained on, instead of
        feeding the full-length signal it never saw during training.
        """
        self.model.eval()
        scores, targets = [], []
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            length = x.shape[-1]
            starts = np.linspace(0, length - crop_len, n_crops).astype(int)
            probs = torch.zeros(x.shape[0], len(self.class_names), device=self.device)
            for start in starts:
                with self._autocast():
                    logits = self.model(x[..., start : start + crop_len])
                probs += torch.sigmoid(logits).float()
            scores.append((probs / len(starts)).cpu().numpy())
            targets.append(y.numpy())
        return np.concatenate(targets), np.concatenate(scores)

    def evaluate(self, loader: DataLoader) -> dict:
        y_true, y_score = self.predict(loader)
        return compute_metrics(y_true, y_score, self.class_names)

    def save_checkpoint(self, path: Path, **extra) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.model.state_dict(), **extra}, path)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> float:
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            metrics = self.evaluate(val_loader)
            if self.scheduler is not None:
                self.scheduler.step()

            macro = metrics["macro_auroc"]
            is_best = macro > self.best_macro_auroc
            if is_best:
                self.best_macro_auroc = macro
                self.save_checkpoint(self.ckpt_dir / "best.pt", epoch=epoch, macro_auroc=macro)

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_macro_auroc": macro,
                "val_macro_f1": metrics["macro_f1"],
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            if self.logger is not None:
                self.logger(record)
            print(
                f"epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                f"val macro-AUROC {macro:.4f} | macro-F1 {metrics['macro_f1']:.4f}"
                + ("  *best" if is_best else "")
            )
        return self.best_macro_auroc
