"""Weights & Biases experiment logging."""

from __future__ import annotations

from typing import Any


class WandbLogger:
    """Thin wrapper over a W&B run, usable as the trainer's per-epoch callback."""

    def __init__(
        self,
        project: str,
        config: dict[str, Any] | None = None,
        group: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        self.run = wandb.init(
            project=project,
            group=group,
            name=name,
            tags=tags,
            notes=notes,
            config=config,
        )

    def __call__(self, record: dict[str, Any]) -> None:
        self._wandb.log(record, step=record.get("epoch"))

    def summary(self, values: dict[str, Any]) -> None:
        self.run.summary.update(values)

    def finish(self) -> None:
        self._wandb.finish()
