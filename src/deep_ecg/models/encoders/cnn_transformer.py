"""CNN + Transformer encoder.

A strided convolutional front-end tokenizes ``(B, 12, T)`` into a short sequence
of feature vectors ``(B, d_model, T')``; a Transformer encoder then applies
self-attention across those tokens to model long-range structure. Returns the
contextualized sequence ``(B, d_model, T')`` so the same head and the
self-supervised objectives apply.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn


class ConvTokenizer(nn.Module):
    """Strided conv-BN-ReLU stack mapping ``(B, 12, T)`` to ``(B, d_model, T')``."""

    def __init__(
        self, in_channels: int, d_model: int, widths: Sequence[int] = (64, 128, 256)
    ) -> None:
        super().__init__()
        channels = [in_channels, *widths, d_model]
        layers: list[nn.Module] = []
        for a, b in zip(channels[:-1], channels[1:], strict=True):
            layers += [
                nn.Conv1d(a, b, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(b),
                nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to ``(B, T, d_model)`` tokens."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CNNTransformerEncoder(nn.Module):
    """Convolutional tokenizer followed by a Transformer encoder."""

    def __init__(
        self,
        in_channels: int = 12,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        conv_widths: Sequence[int] = (64, 128, 256),
    ) -> None:
        super().__init__()
        self.tokenizer = ConvTokenizer(in_channels, d_model, conv_widths)
        self.pos_encoding = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.out_channels = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x).transpose(1, 2)  # (B, T', d_model)
        tokens = self.pos_encoding(tokens)
        tokens = self.transformer(tokens)
        return tokens.transpose(1, 2)  # (B, d_model, T')


def build_cnn_transformer(
    in_channels: int = 12,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    conv_widths: Sequence[int] = (64, 128, 256),
) -> CNNTransformerEncoder:
    return CNNTransformerEncoder(
        in_channels=in_channels,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        conv_widths=conv_widths,
    )
