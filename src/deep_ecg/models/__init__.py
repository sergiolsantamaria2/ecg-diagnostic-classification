"""Model assembly: encoder + task head."""

from __future__ import annotations

from collections.abc import Mapping

from .encoders import build_encoder
from .heads import build_head
from .model import EncoderHeadModel

__all__ = ["EncoderHeadModel", "build_encoder", "build_head", "build_model"]


def build_model(
    encoder: Mapping, head: Mapping, num_classes: int
) -> EncoderHeadModel:
    """Assemble an :class:`EncoderHeadModel` from encoder and head specs.

    ``encoder`` and ``head`` are mappings with a ``name`` and optional ``args``;
    the head's ``in_channels`` is wired from the encoder's ``out_channels``.
    """
    enc = build_encoder(encoder["name"], **(encoder.get("args") or {}))
    task_head = build_head(
        head["name"], in_channels=enc.out_channels, num_classes=num_classes,
        **(head.get("args") or {}),
    )
    return EncoderHeadModel(enc, task_head)
