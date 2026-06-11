"""Plot the cross-source transfer comparison from the W&B ``data_efficiency`` sweep.

Contrasts self-supervised pretraining on PTB-XL itself (in-domain) against
pretraining on the five non-PTB-XL Challenge 2021 databases (cross-source), both
transferred to held-out PTB-XL. For each pretext method a solid curve (in-domain)
and a dashed curve (cross-source) share a colour, so a reader sees at a glance that
the dashed curve tracks the solid one — pretraining on foreign sources transfers as
well as in-domain pretraining. Two panels (linear probe, fine-tune), test
macro-AUROC averaged over subsampling seeds. Reproducible from the logged runs.
Usage: ``uv run python scripts/figures/plot_cross_source.py [arch] [entity/project]``.
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
# method tag -> (label, colour). The cross-source tags reuse the colour of their
# in-domain counterpart so the solid/dashed pairing reads as one method.
METHODS = {
    "masked": ("Masked", "#d1495b"),
    "contrastive": ("Contrastive", "#2e7d32"),
}
CROSS_SUFFIX = "_cs"
PANELS = [("linear_probe", "Linear probe"), ("finetune", "Fine-tune")]
ARCH_LABELS = {"resnet1d": "ResNet1D", "cnn_transformer": "CNN+Transformer"}
SCRATCH_COLOR = "#888888"
REPO_ROOT = Path(__file__).resolve().parents[2]


def collect(path: str, arch: str):
    """Aggregate test macro-AUROC for one architecture.

    Returns ``scratch`` (fraction -> list) and ``ssl`` ((method_tag, regime) ->
    fraction -> list), where ``method_tag`` is e.g. ``masked`` or ``masked_cs``.
    """
    api = wandb.Api()
    runs = api.runs(path, filters={"group": GROUP})
    scratch: dict[float, list[float]] = defaultdict(list)
    ssl: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    method_tags = [m + s for m in METHODS for s in ("", CROSS_SUFFIX)]
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
        method = next((m for m in method_tags if m in tags), None)
        if method is None:
            continue
        regime = "linear_probe" if "linear_probe" in tags else "finetune"
        ssl[(method, regime)][fraction].append(float(auroc))
    return scratch, ssl


def curve(ax, frac_to_vals, color, label, dashed=False):
    if not frac_to_vals:
        return
    fractions = sorted(frac_to_vals)
    means = [float(np.mean(frac_to_vals[f])) for f in fractions]
    stds = [float(np.std(frac_to_vals[f])) for f in fractions]
    ax.errorbar(
        [f * 100 for f in fractions], means, yerr=stds,
        marker="o", markersize=5, capsize=3, color=color, label=label,
        linestyle="--" if dashed else "-", alpha=0.95 if not dashed else 0.85,
    )


def main() -> None:
    arch = sys.argv[1] if len(sys.argv) > 1 else "resnet1d"
    path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PATH
    scratch, ssl = collect(path, arch)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (regime, title) in zip(axes, PANELS, strict=True):
        curve(ax, scratch, SCRATCH_COLOR, "From scratch")
        for method, (label, color) in METHODS.items():
            curve(ax, ssl[(method, regime)], color, f"{label} (in-domain)")
            curve(ax, ssl[(method + CROSS_SUFFIX, regime)], color,
                  f"{label} (cross-source)", dashed=True)
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(["1%", "10%", "100%"])
        ax.set_xlabel("Labeled training data")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Test macro-AUROC")
    fig.suptitle(
        f"Cross-source vs in-domain SSL pretraining, transferred to PTB-XL "
        f"({ARCH_LABELS.get(arch, arch)})"
    )
    fig.tight_layout()

    name = "cross_source.png" if arch == "resnet1d" else f"cross_source_{arch}.png"
    out = REPO_ROOT / "figures" / name
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    def row(method: str, regime: str, frac_to_vals) -> str:
        cells = " ".join(
            f"{np.mean(frac_to_vals[f]):>8.4f}" if frac_to_vals[f] else f"{'-':>8s}"
            for f in (0.01, 0.1, 1.0)
        )
        return f"{method:16s} {regime:13s} {cells}"

    print(f"\n{arch}  {'method':16s} {'regime':13s} {'1%':>8s} {'10%':>8s} {'100%':>8s}")
    print(row("scratch", "--", scratch))
    for (method, regime), frac_to_vals in sorted(ssl.items()):
        print(row(method, regime, frac_to_vals))


if __name__ == "__main__":
    main()
