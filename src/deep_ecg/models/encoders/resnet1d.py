"""1D residual encoder (ResNet/xresnet1d style).

Maps ``(B, 12, T)`` to a feature sequence ``(B, C, T')``. A wide-kernel stem
downsamples once, followed by four residual stages that double the channels and
halve the temporal length. No final pooling — that is the head's job, which
keeps the encoder reusable for self-supervised pretraining.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class BasicBlock1d(nn.Module):
    """Two conv-BN layers with a residual shortcut."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNet1d(nn.Module):
    """Residual 1D CNN encoder."""

    def __init__(
        self,
        in_channels: int = 12,
        widths: Sequence[int] = (64, 128, 256, 512),
        blocks: Sequence[int] = (2, 2, 2, 2),
        kernel_size: int = 5,
        stem_kernel: int = 7,
    ) -> None:
        super().__init__()
        stem_channels = widths[0]
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, stem_kernel, stride=2,
                      padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        stages: list[nn.Module] = []
        channels = stem_channels
        for i, (width, n_blocks) in enumerate(zip(widths, blocks)):
            stride = 1 if i == 0 else 2  # stem already downsampled before stage 0
            stage = [BasicBlock1d(channels, width, kernel_size, stride=stride)]
            stage += [
                BasicBlock1d(width, width, kernel_size) for _ in range(n_blocks - 1)
            ]
            stages.append(nn.Sequential(*stage))
            channels = width
        self.stages = nn.Sequential(*stages)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(x))


def build_resnet1d(
    in_channels: int = 12,
    widths: Sequence[int] = (64, 128, 256, 512),
    blocks: Sequence[int] = (2, 2, 2, 2),
    kernel_size: int = 5,
    stem_kernel: int = 7,
) -> ResNet1d:
    return ResNet1d(
        in_channels=in_channels, widths=widths, blocks=blocks,
        kernel_size=kernel_size, stem_kernel=stem_kernel,
    )
