"""Plot the label-efficiency curve from the W&B ``data_efficiency`` sweep.

Aggregates the sweep runs by regime (from scratch, linear probe, fine-tuning)
and label fraction, averaging test macro-AUROC over subsampling seeds, and writes
the macro-AUROC versus label-fraction figure used in the README. Reads from W&B
so the figure is reproducible from the logged runs rather than copied numbers.
Usage: ``uv run python scripts/plot_data_efficiency.py [entity/project]``.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import wandb

DEFAULT_PATH = "sergiolsantamaria-tu-wien/deep-ecg"
GROUP = "data_efficiency"
REGIMES = {
    "scratch": "From scratch",
    "linear_probe": "Linear probe",
    "finetune": "Fine-tune (SSL)",
}
COLORS = {"scratch": "#888888", "linear_probe": "#d1495b", "finetune": "#2e7d32"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def collect(path: str) -> dict[str, dict[float, list[float]]]:
    """Map regime -> label fraction -> list of test macro-AUROC over seeds."""
    api = wandb.Api()
    runs = api.runs(path, filters={"group": GROUP})
    scores: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        auroc = run.summary.get("test_macro_auroc")
        if auroc is None:  # skip crashed / unfinished runs
            continue
        regime = next((t for t in run.tags if t in REGIMES), None)
        if regime is None:
            continue
        fraction = float(run.config["data"]["label_fraction"])
        scores[regime][fraction].append(float(auroc))
    return scores


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    scores = collect(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for regime, label in REGIMES.items():
        fractions = sorted(scores[regime])
        means = [float(np.mean(scores[regime][f])) for f in fractions]
        stds = [float(np.std(scores[regime][f])) for f in fractions]
        ax.errorbar(
            [f * 100 for f in fractions],
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            color=COLORS[regime],
            label=label,
        )

    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["1%", "10%", "100%"])
    ax.set_xlabel("Labeled training data")
    ax.set_ylabel("Test macro-AUROC")
    ax.set_title("Label efficiency: SSL pretraining vs. from scratch (ResNet1D)")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    out = REPO_ROOT / "docs" / "data_efficiency.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # Console table for the README.
    print(f"\n{'regime':14s} {'1%':>16s} {'10%':>16s} {'100%':>8s}")
    for regime, label in REGIMES.items():
        cells = []
        for frac in (0.01, 0.1):
            vals = scores[regime][frac]
            cells.append(f"{np.mean(vals):.4f}±{np.std(vals):.4f}")
        full = np.mean(scores[regime][1.0])
        print(f"{label:14s} {cells[0]:>16s} {cells[1]:>16s} {full:>8.4f}")


if __name__ == "__main__":
    main()
