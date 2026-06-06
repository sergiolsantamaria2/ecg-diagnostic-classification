"""Losses for self-supervised pretext tasks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Normalized temperature-scaled cross entropy (SimCLR / InfoNCE).

    ``z1`` and ``z2`` are ``(B, d)`` projections of the two views of each record.
    Each view's positive is its counterpart in the other view; every other view
    in the batch is a negative. Returns the mean loss over all ``2B`` views.
    """
    batch = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)  # (2B, d)
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))  # exclude self-similarity
    targets = (torch.arange(2 * batch, device=z.device) + batch) % (2 * batch)
    return F.cross_entropy(sim, targets)
