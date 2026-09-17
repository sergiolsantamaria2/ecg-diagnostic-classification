"""Self-supervised pretext pieces: block masking and the two losses."""

from __future__ import annotations

import pytest
import torch

from deep_ecg.ssl.losses import masked_reconstruction_loss, nt_xent
from deep_ecg.ssl.masked import temporal_block_mask

CPU = torch.device("cpu")


def test_block_mask_ratio_and_whole_spans():
    mask = temporal_block_mask(batch_size=4, length=1000, span=50, mask_ratio=0.5, device=CPU)
    assert mask.shape == (4, 1, 1000)
    assert torch.all(mask.mean(dim=2) == 0.5)  # exactly 10 of the 20 blocks
    blocks = mask.view(4, 20, 50)
    assert torch.all(blocks.amin(dim=2) == blocks.amax(dim=2))  # spans masked whole


def test_block_mask_leaves_trailing_samples_visible():
    mask = temporal_block_mask(batch_size=2, length=1030, span=50, mask_ratio=0.5, device=CPU)
    assert mask[..., 1000:].sum() == 0


def test_masked_loss_scores_masked_positions_only():
    target = torch.zeros(2, 12, 100)
    recon = torch.ones(2, 12, 100)
    mask = torch.zeros(2, 1, 100)
    mask[..., :50] = 1
    assert masked_reconstruction_loss(recon, target, mask).item() == pytest.approx(1.0)
    recon[..., 50:] = 100.0  # errors on visible positions must not count
    assert masked_reconstruction_loss(recon, target, mask).item() == pytest.approx(1.0)


def test_nt_xent_prefers_matching_views():
    torch.manual_seed(0)
    z1 = torch.randn(16, 8)
    aligned = nt_xent(z1, z1.clone(), temperature=0.1)
    unrelated = nt_xent(z1, torch.randn(16, 8), temperature=0.1)
    assert torch.isfinite(aligned)
    assert aligned < unrelated
