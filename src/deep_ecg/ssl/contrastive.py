"""Contrastive signal representation learning (CLOCS-style) pretext task.

Two augmented temporal crops of the same record form a positive pair; crops of
other records in the batch are negatives. The shared Phase-1 encoder is reused
untouched and topped with a projection head; an InfoNCE (NT-Xent) loss pulls the
two views of a record together and pushes different records apart. The projection
head is discarded after pretraining.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
import torch.nn as nn

from .losses import nt_xent


def build_contrastive(encoder: nn.Module, cfg: Mapping) -> nn.Module:
    """Assemble the contrastive model: the encoder plus a projection head."""
    from ..models import EncoderHeadModel
    from ..models.heads import build_head

    projector = build_head(
        "projection",
        in_channels=encoder.out_channels,
        **(cfg["projector"].get("args") or {}),
    )
    return EncoderHeadModel(encoder, projector)


def make_contrastive_loss_step(
    cfg: Mapping,
) -> Callable[[nn.Module, Sequence[torch.Tensor]], torch.Tensor]:
    """Build the ``(model, batch) -> loss`` step, capturing the temperature."""
    temperature = cfg["temperature"]

    def step(model: nn.Module, batch: Sequence[torch.Tensor]) -> torch.Tensor:
        view1, view2 = batch
        return nt_xent(model(view1), model(view2), temperature)

    return step
