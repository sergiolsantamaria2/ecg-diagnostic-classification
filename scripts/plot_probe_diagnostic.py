"""Plot the pretraining diagnostic: downstream probe vs. pretext loss over epochs.

For each pretext method, overlays the online linear-probe macro-AUROC (downstream
proxy) and the validation pretext loss across pretraining epochs. It shows that
masked reconstruction tracks the probe monotonically, whereas the contrastive
objective keeps improving its loss while the probe peaks early and degrades —
motivating checkpoint selection by the probe rather than the pretext loss. Reads
the logged W&B history.
Usage: ``uv run python scripts/plot_probe_diagnostic.py [entity/project]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

import wandb

DEFAULT_PATH = "sergiolsantamaria-tu-wien/deep-ecg"
RUNS = [("masked", "masked_resnet1d_probe"), ("contrastive", "contrastive_resnet1d_probe")]
PROBE_COLOR = "#2e7d32"
LOSS_COLOR = "#888888"
REPO_ROOT = Path(__file__).resolve().parents[1]


def latest_run(api, path: str, name: str):
    """Most recent pretraining run with the given display name."""
    runs = [r for r in api.runs(path, filters={"group": "pretrain"}) if r.name == name]
    return max(runs, key=lambda r: r.created_at)


def history(run):
    epochs, probe, loss = [], [], []
    for row in run.scan_history(keys=["epoch", "val_loss", "probe_auroc"]):
        if row.get("val_loss") is not None:
            epochs.append(row["epoch"])
            loss.append(row["val_loss"])
        if row.get("probe_auroc") is not None:
            probe.append((row["epoch"], row["probe_auroc"]))
    return epochs, loss, probe


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    api = wandb.Api()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (method, name) in zip(axes, RUNS, strict=True):
        epochs, loss, probe = history(latest_run(api, path, name))
        pe, pv = zip(*probe, strict=True)
        peak = pe[pv.index(max(pv))]

        ax.plot(pe, pv, marker="o", color=PROBE_COLOR, label="Linear-probe AUROC")
        ax.axvline(peak, color=PROBE_COLOR, linestyle="--", alpha=0.5)
        ax.set_xlabel("Pretraining epoch")
        ax.set_ylabel("Linear-probe macro-AUROC", color=PROBE_COLOR)
        ax.tick_params(axis="y", labelcolor=PROBE_COLOR)
        ax.set_title(f"{method.capitalize()} (probe peak at epoch {peak})")
        ax.grid(True, linestyle=":", alpha=0.5)

        ax2 = ax.twinx()
        ax2.plot(epochs, loss, color=LOSS_COLOR, alpha=0.8, label="Pretext val loss")
        ax2.set_ylabel("Pretext validation loss", color=LOSS_COLOR)
        ax2.tick_params(axis="y", labelcolor=LOSS_COLOR)

    fig.suptitle("Pretraining diagnostic: downstream probe vs. pretext loss (ResNet1D)")
    fig.tight_layout()
    out = REPO_ROOT / "docs" / "probe_diagnostic.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
