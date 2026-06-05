"""Torch dataset over an :class:`ECGSource`, split by official folds."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .sources.base import ECGSource
from .transforms import Compose, Standardize, Transform, build_augmentations

DEFAULT_TRAIN_FOLDS = (1, 2, 3, 4, 5, 6, 7, 8)
DEFAULT_VAL_FOLD = 9
DEFAULT_TEST_FOLD = 10


class ECGDataset(Dataset):
    """Indexable view over a subset of an :class:`ECGSource`."""

    def __init__(
        self, source: ECGSource, indices: Sequence[int], transform: Transform | None = None
    ) -> None:
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[i])
        signal = self.source.get_signal(idx)
        if self.transform is not None:
            signal = self.transform(signal)
        label = self.source.get_labels(idx)
        return torch.from_numpy(np.ascontiguousarray(signal)), torch.from_numpy(label)


def split_indices_by_fold(
    source: ECGSource,
    train_folds: Sequence[int] = DEFAULT_TRAIN_FOLDS,
    val_fold: int = DEFAULT_VAL_FOLD,
    test_fold: int = DEFAULT_TEST_FOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test index arrays based on the official folds."""
    folds = np.fromiter((source.get_fold(i) for i in range(len(source))), dtype=int)
    train = np.flatnonzero(np.isin(folds, train_folds))
    val = np.flatnonzero(folds == val_fold)
    test = np.flatnonzero(folds == test_fold)
    return train, val, test


def fit_lead_stats(
    source: ECGSource, indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-lead mean and std over ``indices`` (one pass, no leakage)."""
    n_leads = source.get_signal(int(indices[0])).shape[0]
    total = np.zeros(n_leads, dtype=np.float64)
    total_sq = np.zeros(n_leads, dtype=np.float64)
    count = 0
    for i in indices:
        x = source.get_signal(int(i)).astype(np.float64)
        total += x.sum(axis=1)
        total_sq += np.square(x).sum(axis=1)
        count += x.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


def build_datasets(
    source: ECGSource,
    augmentations: dict | None = None,
    train_folds: Sequence[int] = DEFAULT_TRAIN_FOLDS,
    val_fold: int = DEFAULT_VAL_FOLD,
    test_fold: int = DEFAULT_TEST_FOLD,
) -> tuple[ECGDataset, ECGDataset, ECGDataset, dict[str, np.ndarray]]:
    """Build train/val/test datasets with the standardizer fit on train.

    Augmentations apply to the training set only; validation and test see the
    deterministic standardized signal. Returns the three datasets and the
    fitted lead statistics (the reproducibility artifact).
    """
    train_idx, val_idx, test_idx = split_indices_by_fold(
        source, train_folds, val_fold, test_fold
    )
    mean, std = fit_lead_stats(source, train_idx)
    standardize = Standardize(mean, std)
    train_tf = Compose([standardize, *build_augmentations(augmentations)])
    eval_tf = Compose([standardize])
    return (
        ECGDataset(source, train_idx, train_tf),
        ECGDataset(source, val_idx, eval_tf),
        ECGDataset(source, test_idx, eval_tf),
        {"mean": mean, "std": std},
    )
