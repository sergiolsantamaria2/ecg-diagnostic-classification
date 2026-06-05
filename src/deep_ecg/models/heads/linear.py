"""Linear classification head."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """Global average pooling over time followed by a linear classifier.

    Operates on an encoder feature sequence ``(B, C, T')`` and returns raw
    logits ``(B, num_classes)`` (the multi-label loss applies the sigmoid).
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(features).squeeze(-1)
        return self.fc(pooled)


def build_linear(in_channels: int, num_classes: int) -> LinearHead:
    return LinearHead(in_channels=in_channels, num_classes=num_classes)
