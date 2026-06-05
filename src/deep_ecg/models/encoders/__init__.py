"""ECG encoders. Each returns a feature sequence ``(B, C, T')``."""

from __future__ import annotations

import torch.nn as nn


def build_encoder(name: str, **kwargs) -> nn.Module:
    """Construct an encoder by name. Must expose an ``out_channels`` attribute."""
    if name == "simple":
        from .simple import build_simple

        return build_simple(**kwargs)
    raise ValueError(f"unknown encoder: {name!r}")
