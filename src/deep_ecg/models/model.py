"""Encoder + task head composition."""

from __future__ import annotations

import torch
import torch.nn as nn


class EncoderHeadModel(nn.Module):
    """An encoder producing a feature sequence, followed by a task head.

    The encoder maps ``(B, 12, T)`` to a feature sequence ``(B, C, T')``; the
    head maps that to task outputs. Keeping them separate lets the same encoder
    be trained supervised now and pretrained with a self-supervised head later.
    """

    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
