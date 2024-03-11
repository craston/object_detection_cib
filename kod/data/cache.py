from __future__ import annotations

from typing import Optional
from typing import NamedTuple

import pickle
from pathlib import Path
from datetime import datetime

from absl import logging

from rich.table import Table
from rich.console import Console

from kod.core.bbox.boxes import XYXYBoundingBox
from kod.utils.fs import get_default_dataset_cache_dir

from .enums import DatasetName


class ImageMetadata(NamedTuple):
    width: int
    height: int
    num_channels: int
    mime_type: str
    size_bytes: int


class TargetInfo(NamedTuple):
    bounding_box: XYXYBoundingBox
    class_name: str


class SampleInfo(NamedTuple):
    id: str
    image_path: str
    image_metadata: ImageMetadata
    targets: list[TargetInfo]


class DatasetInfo(NamedTuple):
    name: str
    date: datetime
    classes: list[str]
    samples: list[SampleInfo]

    def subset(self, num_samples: int) -> DatasetInfo:
        return DatasetInfo(
            name=self.name,
            date=self.date,
            classes=self.classes[:num_samples],
            samples=self.samples[:num_samples],
        )

    def filter(
        self,
        new_name: str,
        classes_to_include: list[str],
    ) -> DatasetInfo:
        # some sanity checks
        for c in classes_to_include:
            if c not in self.classes:
                raise ValueError(f"{c} is not in the original dataset!")

        filtered_samples: list[SampleInfo] = []
        for s in self.samples:
            filtered_targets: list[TargetInfo] = []
            for t in s.targets:
                if t.class_name in classes_to_include:
                    filtered_targets.append(t)

            if filtered_targets:
                filtered_samples.append(
                    SampleInfo(
                        id=s.id,
                        image_metadata=s.image_metadata,
                        image_path=s.image_path,
                        targets=filtered_targets,
                    )
                )

        logging.info(
            f"Filtering removed {len(self.samples) - len(filtered_samples)} samples..."
        )

        return DatasetInfo(
            name=new_name,
            date=self.date,
            classes=classes_to_include,
            samples=filtered_samples,
        )

    def summarize(self, extra_title: Optional[str] = None):
        console = Console()

        title = self.name
        if extra_title:
            title = f"{title} - [{extra_title}]"

        # find samples that do not have any label
        num_samples_with_no_target = 0
        for sample in self.samples:
            if not sample.targets:
                num_samples_with_no_target = num_samples_with_no_target + 1

        table = Table(title=title, show_header=False)
        table.add_row("Num of classes", str(len(self.classes)))
        table.add_row("Num of samples", str(len(self.samples)))
        table.add_row("Samples w/o target", str(num_samples_with_no_target))

        console.print(table)

        table = Table(
            title="Stats",
            header_style="bold magenta",
        )
        table.add_column("Class")
        table.add_column("Instance Count")
        for class_name, count in self.get_instance_count().items():
            table.add_row(class_name, str(count))

        console = Console()
        console.print(table)

    def get_instance_count(self) -> dict[str, int]:
        class_instance_stats: dict[str, int] = {}

        # to ensure the order of the classes is the same as the order of the
        # classes in the dataset, I am initializing it
        for c in self.classes:
            class_instance_stats[c] = 0

        for sample in self.samples:
            if not sample.targets:
                continue

            for target in sample.targets:
                class_name = target.class_name
                class_instance_stats[class_name] = class_instance_stats[class_name] + 1

        return class_instance_stats


def deserialize_cached_dataset(
    dataset: DatasetName,
    split: str,
    cache_dir: Optional[Path] = None,
) -> DatasetInfo:
    if split not in ("train", "validation"):
        raise ValueError("split can only be train or validation")

    if cache_dir is None:
        cache_dir = get_default_dataset_cache_dir()

    cache_file = cache_dir.joinpath(f"kod-{dataset.value}-{split}.pkl")

    with open(cache_file, "rb") as fp:
        dataset_info: DatasetInfo = pickle.load(fp)

    return dataset_info
