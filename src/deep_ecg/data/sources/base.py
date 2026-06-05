"""Common interface for ECG databases.

An ``ECGSource`` harmonizes a database to a single format — canonical 12-lead
order, fixed sampling rate and duration — and exposes signals, multi-hot labels
and the official fold assignment. PTB-XL implements it now; MIMIC-IV-ECG and the
PhysioNet/CinC 2021 databases plug in later as additional sources without
touching the rest of the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Standard 12-lead order. Every source returns signals in this lead order so a
# model trained on one database sees the same channel layout on another.
CANONICAL_LEADS: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)


class ECGSource(ABC):
    """A read-only, indexable view over an ECG database."""

    sampling_rate: int
    classes: tuple[str, ...]

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def get_signal(self, index: int) -> np.ndarray:
        """Return the ECG as a ``(12, T)`` float32 array in canonical lead order."""

    @abstractmethod
    def get_labels(self, index: int) -> np.ndarray:
        """Return the multi-hot target as a ``(num_classes,)`` float32 array."""

    @abstractmethod
    def get_fold(self, index: int) -> int:
        """Return the official split fold for ``index``."""

    @property
    def num_classes(self) -> int:
        return len(self.classes)
