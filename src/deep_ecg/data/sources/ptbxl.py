"""PTB-XL source adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb

from ..harmonize import reorder_to_canonical
from ..labels import SUPERCLASSES, build_label_table, multihot
from .base import CANONICAL_LEADS, ECGSource


class PTBXLSource(ECGSource):
    """PTB-XL as an :class:`ECGSource`.

    Signals are loaded once and cached to ``<data_dir>/processed`` as a single
    memory-mapped array, so epochs index into RAM/disk instead of re-parsing
    ~21k WFDB records every time. Records without any diagnostic superclass are
    dropped by default, matching the standard superclass benchmark setup.
    """

    classes = SUPERCLASSES

    def __init__(
        self,
        root: str | Path,
        sampling_rate: int = 100,
        drop_unlabeled: bool = True,
        cache: bool = True,
    ) -> None:
        if sampling_rate not in (100, 500):
            raise ValueError("sampling_rate must be 100 or 500")
        self.root = Path(root)
        self.sampling_rate = sampling_rate

        table = build_label_table(
            self.root / "ptbxl_database.csv", self.root / "scp_statements.csv"
        ).sort_index()

        filename_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
        self._filenames = table[filename_col].to_numpy()
        self._ecg_ids = table.index.to_numpy()
        self._labels = np.stack([multihot(s) for s in table["superclasses"]])
        self._folds = table["strat_fold"].to_numpy().astype(int)
        self._signals = self._load_signals(cache)

        mask = self._labels.any(axis=1) if drop_unlabeled else np.ones(len(table), bool)
        self._index = np.flatnonzero(mask)

    # -- ECGSource interface ------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def get_signal(self, index: int) -> np.ndarray:
        return np.asarray(self._signals[self._index[index]], dtype=np.float32)

    def get_labels(self, index: int) -> np.ndarray:
        return self._labels[self._index[index]]

    def get_fold(self, index: int) -> int:
        return int(self._folds[self._index[index]])

    # -- signal loading / caching ------------------------------------------
    def _load_signals(self, cache: bool) -> np.ndarray:
        cache_dir = self.root.parents[1] / "processed"
        sig_path = cache_dir / f"ptbxl_signals_{self.sampling_rate}hz.npy"
        ids_path = cache_dir / f"ptbxl_signals_{self.sampling_rate}hz_ids.npy"

        if cache and sig_path.exists() and ids_path.exists():
            if np.array_equal(np.load(ids_path), self._ecg_ids):
                return np.load(sig_path, mmap_mode="r")

        signals = self._read_all_wfdb()
        if cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(sig_path, signals)
            np.save(ids_path, self._ecg_ids)
            return np.load(sig_path, mmap_mode="r")
        return signals

    def _read_all_wfdb(self) -> np.ndarray:
        n = len(self._filenames)
        first, fields = wfdb.rdsamp(str(self.root / self._filenames[0]))
        n_samples = first.shape[0]
        signals = np.empty((n, len(CANONICAL_LEADS), n_samples), dtype=np.float32)
        signals[0] = reorder_to_canonical(first, fields["sig_name"]).T
        for i in range(1, n):
            sig, fields = wfdb.rdsamp(str(self.root / self._filenames[i]))
            signals[i] = reorder_to_canonical(sig, fields["sig_name"]).T
            if (i + 1) % 2000 == 0:
                print(f"  loaded {i + 1}/{n} records", flush=True)
        return signals
