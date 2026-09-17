"""Encoder/head assembly: shapes, the encoder's sequence output, and freezing."""

from __future__ import annotations

import pytest
import torch

from deep_ecg.data.labels import SUPERCLASSES
from deep_ecg.models import build_model, freeze_encoder

# Small configurations of every encoder so the tests run in seconds on CPU.
ENCODERS = {
    "simple": {"widths": [8, 16]},
    "resnet1d": {"widths": [8, 16, 32, 32], "blocks": [1, 1, 1, 1], "kernel_size": 5},
    "cnn_transformer": {
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "conv_widths": [8, 16],
    },
}


@pytest.mark.parametrize("name", ENCODERS)
def test_forward_shapes(name):
    model = build_model(
        {"name": name, "args": ENCODERS[name]}, {"name": "linear"}, num_classes=len(SUPERCLASSES)
    ).eval()
    x = torch.randn(2, 12, 1000)
    with torch.no_grad():
        features = model.encoder(x)
        logits = model(x)
    # The encoder keeps a (B, C, T') sequence so SSL heads can attach to it.
    assert features.ndim == 3
    assert features.shape[:2] == (2, model.encoder.out_channels)
    assert logits.shape == (2, len(SUPERCLASSES))


def test_freeze_encoder_keeps_head_trainable():
    model = build_model({"name": "simple", "args": {"widths": [8]}}, {"name": "linear"}, 5)
    freeze_encoder(model)
    model.train()  # what the trainer does every epoch
    assert not model.encoder.training
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
