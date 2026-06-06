"""Self-supervised pretext tasks for ECG encoder pretraining.

Each method wraps the shared encoder with a task-specific head and objective and
lives in its own module exposing a ``build_*`` assembler and a loss step.
``build_pretext`` and ``build_loss_step`` dispatch by name, so adding a method is
a new file plus one branch here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
import torch.nn as nn

from .losses import masked_reconstruction_loss, nt_xent

__all__ = ["build_loss_step", "build_pretext", "masked_reconstruction_loss", "nt_xent"]


def build_pretext(name: str, encoder: nn.Module, cfg: Mapping) -> nn.Module:
    """Wrap ``encoder`` in a pretext model by name.

    The encoder is built by the caller and injected so the same instance can be
    lifted out for downstream fine-tuning. ``cfg`` is the method's config block.
    """
    if name == "masked":
        from .masked import build_masked

        return build_masked(encoder, cfg)
    if name == "contrastive":
        from .contrastive import build_contrastive

        return build_contrastive(encoder, cfg)
    raise ValueError(f"unknown pretext task: {name!r}")


def build_loss_step(
    name: str, cfg: Mapping
) -> Callable[[nn.Module, Sequence[torch.Tensor]], torch.Tensor]:
    """Return the ``(model, batch) -> loss`` step for a pretext task.

    Decouples the pretraining loop from the method: each task owns how its model
    output maps to a scalar loss, so the trainer stays method-agnostic.
    """
    if name == "masked":
        from .masked import masked_loss_step

        return masked_loss_step
    if name == "contrastive":
        from .contrastive import make_contrastive_loss_step

        return make_contrastive_loss_step(cfg)
    raise ValueError(f"unknown pretext task: {name!r}")
