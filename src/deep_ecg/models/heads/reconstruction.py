"""Reconstruction head for masked self-supervised pretraining.

A lightweight decoder mapping an encoder feature sequence ``(B, C, T')`` back to
a signal ``(B, 12, T'')``. Following the masked-autoencoder principle the decoder
is intentionally small and discarded after pretraining — the asset is the
encoder. Temporal length is recovered with strided transposed convolutions; the
exact input length is restored by the caller (the masked autoencoder).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionHead(nn.Module):
    """Transposed-convolution decoder from features to a multi-lead signal.

    A 1x1 convolution projects the encoder channels to ``width``; ``n_upsample``
    transposed-conv blocks each double the temporal length at constant width; a
    final 1x1 convolution maps to ``out_channels`` leads.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 12,
        n_upsample: int = 5,
        width: int = 256,
    ) -> None:
        super().__init__()
        self.project = nn.Conv1d(in_channels, width, kernel_size=1)
        blocks: list[nn.Module] = []
        for _ in range(n_upsample):
            blocks += [
                nn.ConvTranspose1d(width, width, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(width),
                nn.GELU(),
            ]
        self.upsample = nn.Sequential(*blocks)
        self.to_signal = nn.Conv1d(width, out_channels, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.to_signal(self.upsample(self.project(features)))


def build_reconstruction(
    in_channels: int,
    out_channels: int = 12,
    n_upsample: int = 5,
    width: int = 256,
) -> ReconstructionHead:
    return ReconstructionHead(
        in_channels=in_channels,
        out_channels=out_channels,
        n_upsample=n_upsample,
        width=width,
    )
