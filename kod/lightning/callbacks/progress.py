from __future__ import annotations

from typing import Any

from rich.table import Table
from rich.console import Console


import lightning.pytorch as LP
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT


class ProgressDisplayCallback(Callback):
    def __init__(
        self,
        progress_interval: int,
        metrics: list[str],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._progress_interval = progress_interval
        self._metrics_to_display = metrics

    def on_train_batch_end(
        self,
        trainer: LP.Trainer,
        pl_module: LP.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if (
            batch_idx % self._progress_interval != 0
            or len(self._metrics_to_display) == 0
        ):
            return

        to_log: dict[str, str] = {}
        for m in self._metrics_to_display:
            to_log[m] = str(trainer.callback_metrics[m].item())

        table = Table(title="Metrics", show_header=False, show_lines=True)

        table.add_row("batch idx", str(batch_idx))
        for k, v in to_log.items():
            table.add_row(k, str(v))

        console = Console()
        console.print(table)
