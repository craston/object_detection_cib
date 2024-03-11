from __future__ import annotations

from pathlib import Path

from typing import Any

from collections import Counter
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt

import lightning.pytorch as LP
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from kod.test_utils.inspect_sampler import plot_and_save_instances_per_class

plt.rcParams.update({"font.size": 18})


class SamplerDebug(Callback):
    def __init__(
        self,
        plot_title: str = "Instances per class",
        sampler_output_dir: str = "./sampler_debug",
    ):
        super().__init__()
        self.plot_title = plot_title
        self.instances_per_class = {}
        self.output_dir = Path(sampler_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = []
        self.idx_class = {}

    def on_train_epoch_start(
        self,
        trainer: LP.Trainer,
        pl_module: LP.LightningModule,
    ) -> None:
        self.instances_per_class = {
            c: [] for c in trainer.datamodule.train_dataset_info.classes
        }
        self.class_names = trainer.datamodule.train_dataset_info.classes
        self.idx_class = {i: c for i, c in enumerate(self.class_names)}

    def on_train_batch_end(
        self,
        trainer: LP.Trainer,
        pl_module: LP.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        detections = batch[1]
        labels = []

        for d in detections:
            labels.extend(d.labels.tolist())

        counter = Counter(labels)
        for cls, count in counter.items():
            self.instances_per_class[self.idx_class[cls]].append(count)

    def on_train_epoch_end(
        self,
        trainer: LP.Trainer,
        pl_module: LP.LightningModule,
    ) -> None:
        # Create the bar plot for the instances per class and save json
        plot_and_save_instances_per_class(
            self.instances_per_class,
            self.class_names,
            self.plot_title,
            self.output_dir,
            batch_size=trainer.datamodule.batch_size,
            epoch=trainer.current_epoch,
            num_workers=trainer.datamodule.num_workers,
        )
