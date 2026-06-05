"""PTB-XL label engineering: SCP codes -> 5 diagnostic superclasses.

Each PTB-XL record carries a dict of SCP codes (code -> likelihood). The
diagnostic statements among them map to one of five superclasses via
``scp_statements.csv``. A record's target is the multi-hot union of the
superclasses of its diagnostic codes (multi-label: several may co-occur).
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

SUPERCLASSES: tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")
_INDEX = {name: i for i, name in enumerate(SUPERCLASSES)}


def load_superclass_map(scp_statements_path: str | Path) -> dict[str, str]:
    """Map each diagnostic SCP code to its superclass (``diagnostic_class``)."""
    df = pd.read_csv(scp_statements_path, index_col=0)
    df = df[df["diagnostic"] == 1]
    return df["diagnostic_class"].to_dict()


def aggregate_superclasses(
    scp_codes: Mapping[str, float], code_to_super: Mapping[str, str]
) -> list[str]:
    """Sorted set of superclasses present among a record's diagnostic codes."""
    present = {code_to_super[c] for c in scp_codes if c in code_to_super}
    return sorted(present)


def multihot(superclasses: Iterable[str]) -> np.ndarray:
    """Encode a set of superclass names as a (5,) multi-hot float32 vector."""
    vec = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    for name in superclasses:
        vec[_INDEX[name]] = 1.0
    return vec


def build_label_table(database_path: str | Path, scp_statements_path: str | Path) -> pd.DataFrame:
    """Load ``ptbxl_database.csv`` augmented with parsed codes and superclasses.

    Returns a frame indexed by ``ecg_id`` with the original columns plus
    ``scp_codes`` (parsed dict) and ``superclasses`` (list of superclass names).
    """
    df = pd.read_csv(database_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    code_to_super = load_superclass_map(scp_statements_path)
    df["superclasses"] = df["scp_codes"].apply(
        lambda codes: aggregate_superclasses(codes, code_to_super)
    )
    return df
