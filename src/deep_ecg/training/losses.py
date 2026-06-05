"""Loss functions for multi-label classification."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_loss(name: str = "bce", pos_weight: torch.Tensor | None = None) -> nn.Module:
    """Construct a loss by name.

    ``bce`` is binary cross-entropy with logits, the standard multi-label loss;
    ``pos_weight`` (a per-class tensor) optionally up-weights positives to offset
    class imbalance.
    """
    if name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    raise ValueError(f"unknown loss: {name!r}")
