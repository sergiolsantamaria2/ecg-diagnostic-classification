"""Task heads mapping encoder features to task outputs."""

from __future__ import annotations

import torch.nn as nn


def build_head(name: str, **kwargs) -> nn.Module:
    """Construct a task head by name."""
    if name == "linear":
        from .linear import build_linear

        return build_linear(**kwargs)
    if name == "reconstruction":
        from .reconstruction import build_reconstruction

        return build_reconstruction(**kwargs)
    if name == "projection":
        from .projection import build_projection

        return build_projection(**kwargs)
    raise ValueError(f"unknown head: {name!r}")
