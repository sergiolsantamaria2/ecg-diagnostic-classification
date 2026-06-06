"""Torch dataset over an :class:`ECGSource`, split by official folds."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .sources.base import ECGSource
from .transforms import Compose, RandomCrop, Standardize, Transform, build_augmentations

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


class TwoViewDataset(Dataset):
    """Yield two independently transformed views of each record.

    Used for contrastive pretraining: applying the same stochastic view
    transform (crop + augmentations) twice produces a positive pair from one
    record. Labels are not returned — the objective is self-supervised.
    """

    def __init__(
        self, source: ECGSource, indices: Sequence[int], view_transform: Transform
    ) -> None:
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)
        self.view_transform = view_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = self.source.get_signal(int(self.indices[i]))
        view1 = self.view_transform(signal)
        view2 = self.view_transform(signal)
        return (
            torch.from_numpy(np.ascontiguousarray(view1)),
            torch.from_numpy(np.ascontiguousarray(view2)),
        )


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


def fit_lead_stats(source: ECGSource, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
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


def subsample_indices(
    indices: np.ndarray, labels: np.ndarray, fraction: float, seed: int
) -> np.ndarray:
    """Random subset of ``indices`` keeping a ``fraction`` of them.

    Selection is reproducible from ``seed``. Every class is guaranteed at least
    one positive: any class absent from the random draw has one of its positives
    swapped in (replacing a random non-guaranteed pick), so the data-efficiency
    regimes never train without a class present.
    """
    if fraction >= 1.0:
        return indices
    rng = np.random.default_rng(seed)
    n = max(1, round(len(indices) * fraction))
    perm = rng.permutation(len(indices))
    chosen = set(perm[:n].tolist())
    guaranteed: set[int] = set()
    for c in range(labels.shape[1]):
        if any(labels[i, c] > 0 for i in chosen):
            continue
        donors = [j for j in range(len(indices)) if labels[j, c] > 0 and j not in chosen]
        if not donors:
            continue
        removable = list(chosen - guaranteed)
        if removable:  # keep the subset size fixed by swapping one out
            chosen.discard(int(rng.choice(removable)))
        new = int(rng.choice(donors))
        chosen.add(new)
        guaranteed.add(new)
    return np.sort(indices[np.fromiter(chosen, dtype=np.int64)])


def build_datasets(
    source: ECGSource,
    augmentations: dict | None = None,
    train_folds: Sequence[int] = DEFAULT_TRAIN_FOLDS,
    val_fold: int = DEFAULT_VAL_FOLD,
    test_fold: int = DEFAULT_TEST_FOLD,
    label_fraction: float = 1.0,
    subsample_seed: int = 0,
) -> tuple[ECGDataset, ECGDataset, ECGDataset, dict[str, np.ndarray]]:
    """Build train/val/test datasets with the standardizer fit on train.

    Augmentations apply to the training set only; validation and test see the
    deterministic standardized signal. Lead statistics are fit on the full train
    folds (they need no labels), then the labeled training set is optionally
    subsampled to ``label_fraction`` for the data-efficiency study. Returns the
    three datasets and the fitted lead statistics (the reproducibility artifact).
    """
    train_idx, val_idx, test_idx = split_indices_by_fold(source, train_folds, val_fold, test_fold)
    mean, std = fit_lead_stats(source, train_idx)
    if label_fraction < 1.0:
        train_labels = np.stack([source.get_labels(int(i)) for i in train_idx])
        train_idx = subsample_indices(train_idx, train_labels, label_fraction, subsample_seed)
    standardize = Standardize(mean, std)
    train_tf = Compose([standardize, *build_augmentations(augmentations)])
    eval_tf = Compose([standardize])
    return (
        ECGDataset(source, train_idx, train_tf),
        ECGDataset(source, val_idx, eval_tf),
        ECGDataset(source, test_idx, eval_tf),
        {"mean": mean, "std": std},
    )


def build_contrastive_datasets(
    source: ECGSource,
    crop_len: int,
    augmentations: dict | None = None,
    train_folds: Sequence[int] = DEFAULT_TRAIN_FOLDS,
    val_fold: int = DEFAULT_VAL_FOLD,
) -> tuple[TwoViewDataset, TwoViewDataset, dict[str, np.ndarray]]:
    """Build two-view train/val datasets for contrastive pretraining.

    Each view is a random crop of the standardized signal plus the configured
    augmentations; drawing it twice gives a positive pair. Statistics are fit on
    the train folds. Returns the train and validation datasets and the fitted
    lead statistics.
    """
    train_idx, val_idx, _ = split_indices_by_fold(source, train_folds, val_fold)
    mean, std = fit_lead_stats(source, train_idx)
    view_tf = Compose(
        [Standardize(mean, std), RandomCrop(crop_len), *build_augmentations(augmentations)]
    )
    return (
        TwoViewDataset(source, train_idx, view_tf),
        TwoViewDataset(source, val_idx, view_tf),
        {"mean": mean, "std": std},
    )
