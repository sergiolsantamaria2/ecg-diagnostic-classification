"""Signal preprocessing and augmentation.

Transforms operate on a ``(12, T)`` float32 array. ``Standardize`` is
deterministic and applied to every split; the augmentations are stochastic and
used only at training time. All augmentations are optional toggles so the
baseline runs on the raw standardized signal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

Transform = Callable[[np.ndarray], np.ndarray]


class Compose:
    """Apply a sequence of transforms in order."""

    def __init__(self, transforms: Sequence[Transform]) -> None:
        self.transforms = list(transforms)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x


class Standardize:
    """Per-lead z-score with fixed (training) statistics."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> None:
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1, 1) + eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


# Augmentations draw from the global NumPy RNG; with multiple DataLoader workers
# a per-worker ``worker_init_fn`` reseeds it so the streams are decorrelated.


class RandomTemporalShift:
    """Shift the signal in time by up to ``max_shift`` samples, zero-filling."""

    def __init__(self, max_shift: int) -> None:
        self.max_shift = max_shift

    def __call__(self, x: np.ndarray) -> np.ndarray:
        shift = int(np.random.randint(-self.max_shift, self.max_shift + 1))
        if shift == 0:
            return x
        out = np.zeros_like(x)
        if shift > 0:
            out[:, shift:] = x[:, :-shift]
        else:
            out[:, :shift] = x[:, -shift:]
        return out


class RandomScaling:
    """Multiply the whole signal by a random factor ~ N(1, sigma)."""

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x * np.float32(np.random.normal(1.0, self.sigma))


class RandomLeadMask:
    """Zero out up to ``max_leads`` randomly chosen leads."""

    def __init__(self, max_leads: int) -> None:
        self.max_leads = max_leads

    def __call__(self, x: np.ndarray) -> np.ndarray:
        k = int(np.random.randint(0, self.max_leads + 1))
        if k == 0:
            return x
        x = x.copy()
        leads = np.random.choice(x.shape[0], size=k, replace=False)
        x[leads] = 0.0
        return x


class GaussianNoise:
    """Add zero-mean Gaussian noise with standard deviation ``sigma``."""

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x + np.random.normal(0.0, self.sigma, size=x.shape).astype(np.float32)


_AUGMENTATIONS = {
    "temporal_shift": (RandomTemporalShift, "max_shift"),
    "scaling": (RandomScaling, "sigma"),
    "lead_mask": (RandomLeadMask, "max_leads"),
    "gaussian_noise": (GaussianNoise, "sigma"),
}


def build_augmentations(config: dict | None) -> list[Transform]:
    """Build the list of enabled augmentations from a config mapping.

    ``config`` maps an augmentation name to its parameter value (or ``None`` /
    falsy to disable it), e.g. ``{"gaussian_noise": 0.1, "lead_mask": 2}``.
    """
    if not config:
        return []
    transforms: list[Transform] = []
    for name, value in config.items():
        if not value:
            continue
        if name not in _AUGMENTATIONS:
            raise ValueError(f"unknown augmentation: {name!r}")
        cls, param = _AUGMENTATIONS[name]
        transforms.append(cls(**{param: value}))
    return transforms
