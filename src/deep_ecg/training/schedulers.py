"""Learning-rate schedulers."""

from __future__ import annotations

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def build_scheduler(
    name: str | None, optimizer: Optimizer, **kwargs
) -> LRScheduler | None:
    """Construct an LR scheduler by name (``None``/``"none"`` disables it)."""
    if name in (None, "none"):
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
    raise ValueError(f"unknown scheduler: {name!r}")
