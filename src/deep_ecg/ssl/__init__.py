"""Self-supervised pretext tasks for ECG encoder pretraining.

Each method wraps the shared encoder with a task-specific head and objective.
``build_pretext`` dispatches by name following the per-implementation pattern, so
adding a method (e.g. contrastive) is a new file plus one branch here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
import torch.nn as nn

from .losses import masked_reconstruction_loss

__all__ = ["build_loss_step", "build_pretext", "masked_reconstruction_loss"]


def build_pretext(name: str, encoder: nn.Module, **kwargs) -> nn.Module:
    """Wrap ``encoder`` in a pretext model by name.

    The encoder is built by the caller and injected so the same instance can be
    lifted out for downstream fine-tuning. Remaining ``kwargs`` are method
    hyperparameters (e.g. masking ratio, decoder spec).
    """
    if name == "masked":
        from ..models.heads import build_head
        from .masked import MaskedAutoencoder

        decoder_spec: Mapping = kwargs.pop("decoder")
        decoder = build_head(
            "reconstruction",
            in_channels=encoder.out_channels,
            **(decoder_spec.get("args") or {}),
        )
        return MaskedAutoencoder(encoder, decoder, **kwargs)
    raise ValueError(f"unknown pretext task: {name!r}")


def build_loss_step(name: str) -> Callable[[nn.Module, torch.Tensor], torch.Tensor]:
    """Return the ``(model, x) -> loss`` step for a pretext task.

    Decouples the pretraining loop from the method: each task owns how its model
    output maps to a scalar loss, so the trainer stays method-agnostic.
    """
    if name == "masked":

        def step(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
            recon, mask = model(x)
            return masked_reconstruction_loss(recon, x, mask)

        return step
    raise ValueError(f"unknown pretext task: {name!r}")
