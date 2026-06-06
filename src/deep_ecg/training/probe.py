"""Online linear-probe monitor for self-supervised pretraining.

A cheap downstream proxy: pooled features from the frozen encoder feed a linear
classifier whose validation macro-AUROC tracks the encoder's diagnostic quality
during pretraining. Used to select the pretraining checkpoint and to early-stop,
since the pretext loss is an imperfect proxy for downstream performance. The
global RNG is saved and restored around the probe so it never perturbs the
pretraining trajectory.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..evaluation.metrics import compute_metrics


@contextlib.contextmanager
def frozen_rng():
    """Restore the global torch RNG state on exit (keeps pretraining identical)."""
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@torch.no_grad()
def extract_features(encoder: nn.Module, loader: DataLoader, device: str) -> tuple:
    """Global-average-pooled encoder features and labels over ``loader``."""
    encoder.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        feats.append(encoder(x).mean(dim=-1).float().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def linear_probe_auroc(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: Sequence[str],
    device: str,
    epochs: int = 300,
    lr: float = 1e-2,
    seed: int = 0,
) -> float:
    """Validation macro-AUROC of a linear classifier on frozen pooled features."""
    with frozen_rng():
        torch.manual_seed(seed)
        x_train, y_train = extract_features(encoder, train_loader, device)
        x_val, y_val = extract_features(encoder, val_loader, device)
        x_train, y_train = x_train.to(device), y_train.to(device)

        clf = nn.Linear(x_train.shape[1], len(class_names)).to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(epochs):  # full-batch on precomputed features (cheap)
            optimizer.zero_grad()
            loss_fn(clf(x_train), y_train).backward()
            optimizer.step()

        clf.eval()
        with torch.no_grad():
            scores = torch.sigmoid(clf(x_val.to(device))).cpu().numpy()
    return compute_metrics(y_val.numpy(), scores, class_names)["macro_auroc"]
