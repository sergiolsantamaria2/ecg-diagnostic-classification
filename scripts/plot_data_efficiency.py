"""Plot the label-efficiency curves from the W&B ``data_efficiency`` sweep.

Aggregates the sweep runs for one encoder architecture by pretext method (masked,
contrastive), downstream regime (linear probe, fine-tuning) and label fraction,
averaging test macro-AUROC over subsampling seeds, and writes the two-panel
figure used in the README: one panel per regime, each comparing the two
self-supervised methods against the supervised-from-scratch reference. Reads from
W&B so the figure is reproducible from the logged runs rather than copied numbers.
Usage: ``uv run python scripts/plot_data_efficiency.py [arch] [entity/project]``.
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
METHODS = {"masked": ("Masked SSL", "#d1495b"), "contrastive": ("Contrastive SSL", "#2e7d32")}
PANELS = [("linear_probe", "Linear probe"), ("finetune", "Fine-tune")]
ARCH_LABELS = {"resnet1d": "ResNet1D", "cnn_transformer": "CNN+Transformer"}
SCRATCH_COLOR = "#888888"
REPO_ROOT = Path(__file__).resolve().parents[1]


def collect(path: str, arch: str):
    """Aggregate test macro-AUROC for one architecture.

    Returns ``scratch`` (fraction -> list) and ``ssl`` ((method, regime) ->
    fraction -> list).
    """
    api = wandb.Api()
    runs = api.runs(path, filters={"group": GROUP})
    scratch: dict[float, list[float]] = defaultdict(list)
    ssl: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        auroc = run.summary.get("test_macro_auroc")
        if auroc is None:  # crashed / unfinished
            continue
        if run.config.get("model", {}).get("name") != arch:
            continue
        tags = set(run.tags)
        fraction = float(run.config["data"]["label_fraction"])
        if "scratch" in tags:
            scratch[fraction].append(float(auroc))
            continue
        method = next((m for m in METHODS if m in tags), None)
        if method is None:  # skip other tags (e.g. probe-selection variants)
            continue
        regime = "linear_probe" if "linear_probe" in tags else "finetune"
        ssl[(method, regime)][fraction].append(float(auroc))
    return scratch, ssl


def curve(ax, frac_to_vals, color, label):
    fractions = sorted(frac_to_vals)
    means = [float(np.mean(frac_to_vals[f])) for f in fractions]
    stds = [float(np.std(frac_to_vals[f])) for f in fractions]
    ax.errorbar(
        [f * 100 for f in fractions], means, yerr=stds,
        marker="o", capsize=3, color=color, label=label,
    )


def main() -> None:
    arch = sys.argv[1] if len(sys.argv) > 1 else "resnet1d"
    path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PATH
    scratch, ssl = collect(path, arch)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (regime, title) in zip(axes, PANELS, strict=True):
        curve(ax, scratch, SCRATCH_COLOR, "From scratch")
        for method, (label, color) in METHODS.items():
            curve(ax, ssl[(method, regime)], color, label)
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(["1%", "10%", "100%"])
        ax.set_xlabel("Labeled training data")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend()
    axes[0].set_ylabel("Test macro-AUROC")
    fig.suptitle(f"Label efficiency of self-supervised pretraining ({ARCH_LABELS.get(arch, arch)})")
    fig.tight_layout()

    name = "data_efficiency.png" if arch == "resnet1d" else f"data_efficiency_{arch}.png"
    out = REPO_ROOT / "docs" / name
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    def row(method: str, regime: str, frac_to_vals) -> str:
        cells = " ".join(f"{np.mean(frac_to_vals[f]):>8.4f}" for f in (0.01, 0.1, 1.0))
        return f"{method:12s} {regime:13s} {cells}"

    print(f"\n{arch}  {'method':12s} {'regime':13s} {'1%':>8s} {'10%':>8s} {'100%':>8s}")
    print(row("scratch", "--", scratch))
    for (method, regime), frac_to_vals in sorted(ssl.items()):
        print(row(method, regime, frac_to_vals))


if __name__ == "__main__":
    main()
