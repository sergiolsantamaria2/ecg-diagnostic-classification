"""Minimal convolutional encoder (pipeline smoke test).

A small stack of strided conv-BN-ReLU blocks. Not a competitive architecture —
it exists to validate the end-to-end pipeline before the ResNet1D and
CNN+Transformer encoders are built.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class SimpleConvEncoder(nn.Module):
    """Strided conv blocks mapping ``(B, 12, T)`` to ``(B, C, T')``."""

    def __init__(self, in_channels: int = 12, widths: Sequence[int] = (32, 64, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for width in widths:
            layers += [
                nn.Conv1d(channels, width, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(width),
                nn.ReLU(inplace=True),
            ]
            channels = width
        self.net = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_simple(in_channels: int = 12, widths: Sequence[int] = (32, 64, 128)) -> SimpleConvEncoder:
    return SimpleConvEncoder(in_channels=in_channels, widths=widths)
