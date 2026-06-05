"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: decorrelate each worker's NumPy/Python RNG.

    PyTorch gives each worker a distinct ``initial_seed``; deriving the NumPy and
    Python seeds from it keeps augmentation streams independent across workers
    yet reproducible for a given run seed.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
