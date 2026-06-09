"""PhysioNet/CinC 2021 multi-source adapter.

Pools several Challenge 2021 databases into one unlabeled corpus for cross-source
pretraining. Each database has its own sampling rate and record duration; this
adapter harmonizes every record to the canonical format shared with PTB-XL —
12 leads in canonical order, 100 Hz, a fixed 10 s window — so an encoder pretrained
here transfers to PTB-XL without any input-shape mismatch.

Records carry SNOMED-CT diagnoses, but the cross-source experiment holds out PTB-XL
(consumed natively with its diagnostic labels) and uses these databases purely for
self-supervised pretraining, so labels are not mapped: ``get_labels`` returns zeros.
Each record's source database is exposed for diagnostics, and a deterministic
pseudo-fold (hashed from the record id) yields the train/validation split the
pretraining harness expects.
"""

from __future__ import annotations

import hashlib
from math import gcd
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly

from ..labels import SUPERCLASSES
from .base import CANONICAL_LEADS, ECGSource

# Five non-PTB-XL databases. PTB-XL is held out; PTB and St Petersburg INCART are
# atypical (small / 257 Hz, 30-min) and excluded from the pretraining corpus.
DEFAULT_DATABASES: tuple[str, ...] = (
    "cpsc_2018",
    "cpsc_2018_extra",
    "georgia",
    "chapman_shaoxing",
    "ningbo",
)


class CinC2021Source(ECGSource):
    """Pooled CinC 2021 databases as an unlabeled :class:`ECGSource`.

    Signals are harmonized to ``(12, target_len)`` at ``sampling_rate`` Hz and
    cached per database under ``<data_dir>/processed`` as memory-mapped arrays, so
    a database is parsed once and validating one database never rebuilds the others.
    """

    classes = SUPERCLASSES  # shape only; targets are all-zero (corpus is unlabeled)

    def __init__(
        self,
        root: str | Path,
        databases: tuple[str, ...] = DEFAULT_DATABASES,
        sampling_rate: int = 100,
        duration_s: float = 10.0,
        records_per_db: int | None = None,
        cache: bool = True,
        drop_unlabeled: bool = False,  # accepted for interface parity; ignored
    ) -> None:
        self.root = Path(root)
        self.sampling_rate = sampling_rate
        self.target_len = round(duration_s * sampling_rate)
        self.databases = tuple(databases)

        self._arrays: list[np.ndarray] = []
        self._db_of: list[np.ndarray] = []
        self._stems: list[np.ndarray] = []
        for db in self.databases:
            signals, stems = self._load_database(db, cache)
            if records_per_db is not None and len(stems) > records_per_db:
                signals, stems = signals[:records_per_db], stems[:records_per_db]
            self._arrays.append(signals)
            self._db_of.append(np.full(len(stems), db, dtype=object))
            self._stems.append(stems)

        sizes = [len(a) for a in self._arrays]
        self._offsets = np.cumsum([0, *sizes])
        self.db_index = np.concatenate(self._db_of) if self._db_of else np.array([], dtype=object)
        stems_all = np.concatenate(self._stems) if self._stems else np.array([], dtype=object)
        self._folds = np.fromiter((_pseudo_fold(s) for s in stems_all), dtype=int)

    # -- ECGSource interface ------------------------------------------------
    def __len__(self) -> int:
        return int(self._offsets[-1])

    def get_signal(self, index: int) -> np.ndarray:
        db = int(np.searchsorted(self._offsets, index, side="right") - 1)
        local = index - self._offsets[db]
        return np.asarray(self._arrays[db][local], dtype=np.float32)

    def get_labels(self, index: int) -> np.ndarray:
        return np.zeros(self.num_classes, dtype=np.float32)

    def get_fold(self, index: int) -> int:
        return int(self._folds[index])

    # -- signal loading / caching ------------------------------------------
    def _load_database(self, db: str, cache: bool) -> tuple[np.ndarray, np.ndarray]:
        cache_dir = self.root.parents[1] / "processed"
        tag = f"cinc2021_{db}_{self.sampling_rate}hz_{self.target_len}"
        sig_path = cache_dir / f"{tag}.npy"
        ids_path = cache_dir / f"{tag}_ids.npy"

        if cache and sig_path.exists() and ids_path.exists():
            return np.load(sig_path, mmap_mode="r"), np.load(ids_path, allow_pickle=True)

        db_dir = self.root / db
        stems = sorted(p.stem for p in db_dir.glob("*.hea") if (db_dir / f"{p.stem}.mat").exists())
        if not stems:
            raise FileNotFoundError(f"no records found for {db!r} under {self.root / db}")
        signals = np.empty((len(stems), len(CANONICAL_LEADS), self.target_len), dtype=np.float32)
        for i, stem in enumerate(stems):
            signals[i] = self._read_record(self.root / db / stem)
            if (i + 1) % 5000 == 0:
                print(f"  {db}: harmonized {i + 1}/{len(stems)} records", flush=True)
        ids = np.array(stems, dtype=object)
        if cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(sig_path, signals)
            np.save(ids_path, ids)
            return np.load(sig_path, mmap_mode="r"), ids
        return signals, ids

    def _read_record(self, record_path: Path) -> np.ndarray:
        """Read one WFDB record and harmonize it to ``(12, target_len)``.

        A few CinC records carry non-finite samples (saturated or missing ADC
        values); they are zeroed before resampling so a single bad sample cannot
        poison the polyphase filter — and, downstream, the training loss.
        """
        sig, fields = wfdb.rdsamp(str(record_path))  # (T, n) in physical units
        canonical = self._to_canonical(sig, fields["sig_name"]).T  # (12, T)
        canonical = np.nan_to_num(canonical, nan=0.0, posinf=0.0, neginf=0.0)
        resampled = self._resample(canonical, int(fields["fs"]))
        return self._fit_length(resampled)

    def _resample(self, signal: np.ndarray, orig_fs: int) -> np.ndarray:
        """Polyphase anti-aliased resample of ``(12, T)`` to the target rate."""
        if orig_fs == self.sampling_rate:
            return signal.astype(np.float32, copy=False)
        g = gcd(orig_fs, self.sampling_rate)
        up, down = self.sampling_rate // g, orig_fs // g
        return resample_poly(signal, up, down, axis=1).astype(np.float32)

    def _fit_length(self, signal: np.ndarray) -> np.ndarray:
        """Center-crop or zero-pad ``(12, T)`` to ``target_len`` samples."""
        t = signal.shape[1]
        if t == self.target_len:
            return signal
        if t > self.target_len:
            start = (t - self.target_len) // 2
            return signal[:, start : start + self.target_len]
        out = np.zeros((signal.shape[0], self.target_len), dtype=np.float32)
        start = (self.target_len - t) // 2
        out[:, start : start + t] = signal
        return out

    @staticmethod
    def _to_canonical(signal: np.ndarray, sig_names: list[str]) -> np.ndarray:
        """Reorder ``(T, n)`` signal columns into canonical lead order."""
        col = {name.upper(): i for i, name in enumerate(sig_names)}
        order = [col[lead.upper()] for lead in CANONICAL_LEADS]
        return signal[:, order]


def _pseudo_fold(stem: str) -> int:
    """Deterministic fold in ``1..10`` from a record id (≈90/10 train/val split)."""
    digest = hashlib.md5(stem.encode()).hexdigest()
    return int(digest, 16) % 10 + 1
