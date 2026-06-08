"""ECG source adapters."""

from __future__ import annotations

from .base import CANONICAL_LEADS, ECGSource


def build_source(name: str, **kwargs) -> ECGSource:
    """Construct an :class:`ECGSource` by name."""
    if name == "ptbxl":
        from .ptbxl import PTBXLSource

        return PTBXLSource(**kwargs)
    if name == "cinc2021":
        from .cinc2021 import CinC2021Source

        return CinC2021Source(**kwargs)
    raise ValueError(f"unknown ECG source: {name!r}")


__all__ = ["CANONICAL_LEADS", "ECGSource", "build_source"]
