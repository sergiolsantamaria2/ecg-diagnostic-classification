"""Projection head for contrastive self-supervised pretraining.

Maps an encoder feature sequence ``(B, C, T')`` to a projection vector
``(B, proj_dim)`` via global average pooling and a small MLP (the SimCLR
projector). Used only during contrastive pretraining and discarded afterwards;
the contrastive loss applies the L2 normalization.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Global average pooling followed by a two-layer MLP."""

    def __init__(self, in_channels: int, proj_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(features).squeeze(-1)
        return self.mlp(pooled)


def build_projection(
    in_channels: int, proj_dim: int = 128, hidden_dim: int = 256
) -> ProjectionHead:
    return ProjectionHead(in_channels=in_channels, proj_dim=proj_dim, hidden_dim=hidden_dim)
