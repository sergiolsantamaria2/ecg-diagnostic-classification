"""ECG source adapters."""

from __future__ import annotations

from .base import CANONICAL_LEADS, ECGSource


def build_source(name: str, **kwargs) -> ECGSource:
    """Construct an :class:`ECGSource` by name."""
    if name == "ptbxl":
        from .ptbxl import PTBXLSource

        return PTBXLSource(**kwargs)
    raise ValueError(f"unknown ECG source: {name!r}")


__all__ = ["CANONICAL_LEADS", "ECGSource", "build_source"]
