"""Losses for self-supervised pretext tasks."""

from __future__ import annotations

import torch


def masked_reconstruction_loss(
    recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared error over masked positions only.

    ``mask`` is ``(B, 1, T)`` and broadcasts over leads; the loss averages the
    squared error across masked time steps and all leads.
    """
    se = (recon - target) ** 2
    mask = mask.expand_as(se)
    return (se * mask).sum() / mask.sum().clamp_min(1.0)
