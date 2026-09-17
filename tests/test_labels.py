"""SCP codes -> superclass label engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from deep_ecg.data.labels import aggregate_superclasses, build_label_table, multihot

CODE_TO_SUPER = {"NORM": "NORM", "IMI": "MI", "LVH": "HYP"}


def test_aggregate_ignores_non_diagnostic_codes_and_sorts():
    codes = {"LVH": 100.0, "SR": 0.0, "IMI": 50.0}  # SR is a rhythm code, not diagnostic
    assert aggregate_superclasses(codes, CODE_TO_SUPER) == ["HYP", "MI"]


def test_multihot_positions():
    vec = multihot(["MI", "HYP"])
    assert vec.dtype == np.float32
    assert vec.tolist() == [0.0, 1.0, 0.0, 0.0, 1.0]


def test_build_label_table(tmp_path):
    pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "scp_codes": [
                "{'NORM': 100.0, 'SR': 0.0}",
                "{'IMI': 100.0, 'LVH': 50.0}",
                "{'SR': 0.0}",
            ],
            "strat_fold": [1, 9, 10],
        }
    ).to_csv(tmp_path / "db.csv", index=False)
    pd.DataFrame(
        {
            "code": ["NORM", "IMI", "LVH", "SR"],
            "diagnostic": [1, 1, 1, 0],
            "diagnostic_class": ["NORM", "MI", "HYP", None],
        }
    ).to_csv(tmp_path / "scp.csv", index=False)

    table = build_label_table(tmp_path / "db.csv", tmp_path / "scp.csv")

    assert list(table.index) == [1, 2, 3]
    assert table.loc[1, "scp_codes"] == {"NORM": 100.0, "SR": 0.0}
    assert table.loc[1, "superclasses"] == ["NORM"]
    assert table.loc[2, "superclasses"] == ["HYP", "MI"]
    assert table.loc[3, "superclasses"] == []  # rhythm-only record: no superclass
