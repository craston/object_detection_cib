from __future__ import annotations

from absl import logging

from typing import Any
from typing import Optional

from functools import partial

import json
from pathlib import Path
from collections import Counter

import typer
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import lightning as L

from torch.utils.data import Sampler
from kod.data.enums import DatasetName
from kod.data.mosaic import MosaicAugmentor
from kod.data.sample_reader import SampleReader
from kod.data.samplers import ClassAwareSampler
from kod.data.samplers import RepeatFactorSampler
from kod.nn.networks.yolov5 import Yolov5Network
from kod.data.augmentations.default import AugParams
from kod.data.cache import deserialize_cached_dataset
from kod.lightning.data_module import DetectionDataModule
from kod.data.augmentations.default import TrainSampleAugmentor

plt.rcParams.update({"font.size": 18})

app = typer.Typer()


# Sampler
samplers: dict[str, Sampler] = {
    "class_aware": ClassAwareSampler,
    "repeat_factor": RepeatFactorSampler,
}


def plot_and_save_instances_per_class(
    instances_per_class: dict,
    image_paths: list,
    class_names: list,
    plot_title: str,
    output_path: Path,
    batch_size: int,
    epoch: int = 0,
    num_workers: int = 0,
) -> None:
    fig, ax = plt.subplots(
        len(class_names),
        1,
        figsize=(10, 4 * len(class_names)),
    )
    colors = cm.rainbow(np.linspace(0, 1, len(class_names)))

    for i, (cls, count_list) in enumerate(instances_per_class.items()):
        ax[i].bar(
            range(len(instances_per_class[cls])),
            np.array(count_list) / batch_size,
            color=colors[i],
            label=f"{cls} - {(np.mean(count_list)/batch_size):.2f}",
        )
        ax[i].legend()
        ax[i].set_ylabel(f"avg. instances per image")
        ax[i].set_xlabel("batch index")
        ax[i].set_ylim(0, 2)

    fig.suptitle(plot_title)
    filename = f"{plot_title}_batchSize_{batch_size}"
    filename = f"{filename}_numWorkers_{num_workers}"
    filename = output_path.joinpath(Path(filename))
    plt.tight_layout()
    plt.savefig(
        f"{filename}_{epoch}.png",
    )

    json.dump(
        instances_per_class,
        open(f"{filename}_{epoch}.json", "w"),
    )
    json.dump(
        image_paths,
        open(f"{filename}_image_paths_{epoch}.json", "w"),
    )


@app.command()
def inspect_sampler(
    seed: int = typer.Option(2023, help="Seed"),
    dataset_name: str = typer.Option("voc-toy", help="Daataset name"),
    max_epochs: int = typer.Option(3, help="Number of epochs to run"),
    num_workers: int = typer.Option(0, help="Number of workers"),
    batch_size: int = typer.Option(16, help="Batch size"),
    mosaic: bool = typer.Option(False, help="Use mosaic augmentation"),
    target_image_size: int = typer.Option(416, help="Target image size"),
    plot_title: str = typer.Option("noMosaic_noSampler", help="Plot title"),
    output_dir: Path = typer.Option("./sampler_debug", help="Output directory"),
    sampler_name: Optional[str] = typer.Option(None, help="Sampler name"),
    deepen_factor: float = typer.Option(0.33, help="network's deepen factor"),
    widen_factor: float = typer.Option(0.25, help="network's widen factor"),
    enable_ram_cache: bool = typer.Option(False, help="Enable RAM cache"),
) -> None:
    L.seed_everything(seed, workers=True)

    dataset_name = DatasetName(dataset_name)

    train_dataset_info = deserialize_cached_dataset(dataset_name, split="train")

    sample_reader = SampleReader(
        target_image_size=target_image_size,
        classes=train_dataset_info.classes,
        fake_mode=False,
    )

    if mosaic:
        mosaic_augmentor = MosaicAugmentor(
            target_image_size=target_image_size,
        )
    else:
        mosaic_augmentor = None

    if sampler_name == None:
        sampler = None
    else:
        if sampler_name not in samplers:
            raise ValueError(f"Sampler {sampler_name} not found")

        # this is messy! Need to clean up
        if sampler_name == "repeat_factor":
            sampler = partial(samplers[sampler_name], reduction=None)
        else:
            sampler = partial(samplers[sampler_name])

    data_module = DetectionDataModule(
        batch_size=batch_size,
        enable_ram_cache=enable_ram_cache,
        sample_reader=sample_reader,
        train_dataset_info=train_dataset_info,
        train_data_augmentor=TrainSampleAugmentor(AugParams()),
        num_workers=num_workers,
        mosaic_augmentor=mosaic_augmentor,
        sampler=sampler,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize the network to get exact numbers as that
    # using the sampler_debug callback
    _ = Yolov5Network(
        num_classes=len(train_dataset_info.classes),
        num_anchors_per_cell=3,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    )

    data_module.setup(stage="fit")
    dataloader = data_module.train_dataloader()

    class_names = data_module.train_dataset_info.classes
    idx_class = {i: c for i, c in enumerate(class_names)}

    instances_per_class_avg = {c: [] for c in class_names}
    for epoch in range(max_epochs):
        print(f"Epoch {epoch}")
        # statistics of number of instances per class per batch
        instances_per_class = {c: [] for c in class_names}
        image_paths = []
        for batch in tqdm(dataloader):
            targets = batch[1]
            image_info = batch[2]

            assert len(targets) == len(image_info)

            labels = []

            for target, info in zip(targets, image_info):
                labels.extend(target.labels.tolist())
                if mosaic:
                    continue
                image_paths.append(info.image_path)

            for cls, count in Counter(labels).items():
                instances_per_class[idx_class[cls]].append(count)

        for cls, count_list in instances_per_class.items():
            instances_per_class_avg[cls].append(np.mean(count_list))

        plot_and_save_instances_per_class(
            instances_per_class,
            image_paths,
            class_names,
            plot_title,
            output_path,
            batch_size=data_module.batch_size,
            epoch=epoch,
            num_workers=data_module.num_workers,
        )

    # plot average instances per class
    fig, ax = plt.subplots(
        len(class_names),
        1,
        figsize=(10, 4 * len(class_names)),
    )
    colors = cm.rainbow(np.linspace(0, 1, len(class_names)))
    for i, (cls, count_list) in enumerate(instances_per_class_avg.items()):
        ax[i].bar(
            range(len(instances_per_class_avg[cls])),
            np.array(count_list),
            color=colors[i],
            label=f"{cls} - {np.mean(count_list):.2f}",
        )
        ax[i].legend()
        ax[i].set_ylabel(f"avg. instances per image")
        ax[i].set_xlabel("epoch")

    # save figure
    plt.tight_layout()
    plt.savefig(
        f"Final statistcs of batches.png",
    )
