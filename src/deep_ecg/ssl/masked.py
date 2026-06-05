"""Masked signal reconstruction (masked autoencoding) pretext task.

Contiguous temporal spans are masked across all leads and an encoder-decoder is
trained to reconstruct them. The encoder is the shared Phase-1 encoder, used
untouched; masking happens in input space so no encoder change is required. The
decoder (a reconstruction head) is discarded after pretraining.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def temporal_block_mask(
    batch_size: int, length: int, span: int, mask_ratio: float, device: torch.device
) -> torch.Tensor:
    """Boolean-valued ``(B, 1, length)`` mask, ``1`` where masked.

    Time is partitioned into contiguous blocks of ``span`` samples; a fraction
    ``mask_ratio`` of the blocks is masked per sample, drawn independently. Spans
    are non-overlapping and the realized ratio matches the target exactly. Any
    trailing samples beyond a whole number of blocks stay visible.
    """
    n_blocks = length // span
    n_mask = max(1, round(mask_ratio * n_blocks))
    scores = torch.rand(batch_size, n_blocks, device=device)
    masked_ids = scores.argsort(dim=1)[:, :n_mask]
    block_mask = torch.zeros(batch_size, n_blocks, device=device)
    block_mask.scatter_(1, masked_ids, 1.0)
    mask = block_mask.repeat_interleave(span, dim=1)
    if mask.shape[1] < length:
        pad = torch.zeros(batch_size, length - mask.shape[1], device=device)
        mask = torch.cat([mask, pad], dim=1)
    return mask.unsqueeze(1)


class MaskedAutoencoder(nn.Module):
    """Encoder + reconstruction decoder trained to inpaint masked spans.

    ``forward`` masks the input, encodes the corrupted signal, decodes back to
    the original length and returns the reconstruction together with the mask so
    the loss can be computed on masked positions only. The encoder is kept
    separate (injected) so it can be lifted out for downstream fine-tuning.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        mask_ratio: float = 0.5,
        mask_span: int = 50,
        learnable_token: bool = True,
        in_leads: int = 12,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.mask_span = mask_span
        token = torch.zeros(1, in_leads, 1)
        if learnable_token:
            self.mask_token = nn.Parameter(token)
        else:
            self.register_buffer("mask_token", token)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = x.shape[-1]
        mask = temporal_block_mask(
            x.shape[0], length, self.mask_span, self.mask_ratio, x.device
        )
        x_masked = x * (1.0 - mask) + self.mask_token * mask
        features = self.encoder(x_masked)
        recon = self.decoder(features)
        recon = F.interpolate(recon, size=length, mode="linear", align_corners=False)
        return recon, mask
