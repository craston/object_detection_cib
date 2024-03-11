from __future__ import annotations

import pickle

from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import zipfian
import matplotlib.pyplot as plt

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from fiftyone.core.metadata import ImageMetadata as fiftyOneImageMetadata
from fiftyone.utils.voc import VOCBoundingBox as fiftyOneVOCBoundingBox

from data_gradients.datasets.detection import YoloFormatDetectionDataset
from data_gradients.managers.detection_manager import DetectionAnalysisManager

import typer
from absl import logging
from PIL import Image

from kod.core.bbox.boxes import XYXYBoundingBox
from kod.utils.fs import (
    get_default_datasets_dir,
    get_default_dataset_cache_dir,
)

from .enums import DatasetName
from .cache import DatasetInfo
from .cache import SampleInfo
from .cache import TargetInfo
from .cache import ImageMetadata


def _kod_dataset_name(name: DatasetName, split: str) -> str:
    return f"kod-{name.value}-{split}"


def _delete_dataset(name: DatasetName, split: str):
    ds_name = _kod_dataset_name(name, split)
    if fo.dataset_exists(ds_name):
        fo.delete_dataset(ds_name)


def _dataset_exists(name: DatasetName, split: str) -> bool:
    return fo.dataset_exists(_kod_dataset_name(name, split))


def _load_dataset(name: DatasetName, split: str) -> fo.Dataset:
    # Note - we intentionally do not check if the dataset exists
    # let load_dataset handle that!
    ds_name = _kod_dataset_name(name, split)
    return fo.load_dataset(ds_name)


def _to_cached_sample(sample: fo.Sample) -> SampleInfo:
    targets: list[TargetInfo] = []
    det: fo.Detection
    for det in sample["ground_truth"].detections:
        image_width = sample.metadata.width
        image_height = sample.metadata.height

        voc_box: fiftyOneVOCBoundingBox = fiftyOneVOCBoundingBox.from_detection_format(
            det.bounding_box,
            (image_width, image_height),
        )

        bounding_box = XYXYBoundingBox(
            x_min=voc_box.xmin,
            y_min=voc_box.ymin,
            x_max=voc_box.xmax,
            y_max=voc_box.ymax,
        )

        class_name = det.label

        targets.append(
            TargetInfo(
                bounding_box=bounding_box,
                class_name=class_name,
            )
        )

    assert len(targets) > 0, "We expect atleast one target per sample"

    image_path: str = sample.filepath

    # we remove the machine specify path
    # so that these pickle files can be used
    # on any machine
    image_path = image_path.replace(str(Path.home()) + "/", "")
    sample_meta: fiftyOneImageMetadata = sample.metadata

    return SampleInfo(
        id=sample.id,
        image_path=image_path,
        image_metadata=ImageMetadata(
            width=sample_meta.width,
            height=sample_meta.height,
            num_channels=sample_meta.num_channels,
            size_bytes=sample_meta.size_bytes,
            mime_type=sample_meta.mime_type,
        ),
        targets=targets,
    )

def _get_zipf_distribution(
    num_classes: int, zipf_param: float, num_samples: int
) -> list[int]:
    x = np.arange(1, num_classes + 1)
    zipf_distribution = num_samples * zipfian.pmf(x, zipf_param, num_classes)

    return [int(x) for x in zipf_distribution]


def _get_ds_with_top_classes(
    ds: fo.Dataset, num_classes: int
) -> tuple[fo.Dataset, dict[str, int]]:
    # Considering images less than 10 detections
    ds_view: fo.DatasetView = ds.match(F("ground_truth.detections").length() < 10)

    instance_counts: dict[str, int] = ds_view.count_values(
        "ground_truth.detections.label"
    )
    orig_classes = list(instance_counts.keys())

    # Removing extra classes in original coco dataset
    ds_view = ds_view.filter_labels("ground_truth", F("label").is_in(orig_classes))

    img_counts = {}
    for cls in orig_classes:
        img_counts[cls] = len(
            ds_view.match(F("ground_truth.detections.label").contains(cls))
        )

    top_classes = dict(
        sorted(img_counts.items(), key=lambda x: x[1], reverse=True)[:num_classes]
    )
    ds_view = ds_view.filter_labels(
        "ground_truth", F("label").is_in(list(top_classes.keys()))
    )

    final_dataset: fo.Dataset = ds_view.clone()

    orig_img_counts = _get_image_count(final_dataset, list(top_classes.keys()))
    print("==========================")
    print("original image counts: ", orig_img_counts)
    print("============================")
    return final_dataset, top_classes


def _get_image_count(ds: fo.Dataset, classes: list[str]) -> dict[str, int]:
    img_counts = {}
    for cls in classes:
        img_counts[cls] = len(
            ds.match(F("ground_truth.detections.label").contains(cls))
        )
    return img_counts


def _convert_dataset_to_zipf(
    ds: fo.Dataset, top_classes: dict[str, int], zipf_prob: list[float]
) -> fo.Dataset:
    # make the dataset follow the zipf distribution
    dataset_zipf: fo.Dataset = fo.Dataset()
    sample_indexes = []
    for cls, count in zip(list(top_classes.keys())[::-1], zipf_prob[::-1]):
        print(f"Adding {count} samples of class {cls}")
        tmp: fo.DatasetView = ds.match(F("ground_truth.detections.label").contains(cls))

        # remove the samples that are already added
        tmp = tmp.exclude(sample_indexes)

        if len(dataset_zipf) == 0:
            tmp = tmp.limit(count)
            dataset_zipf = tmp.clone()
        else:
            curr_img_count = _get_image_count(dataset_zipf, list(top_classes.keys()))
            tmp = tmp.limit(count - curr_img_count[cls])
            dataset_zipf.merge_samples(tmp)
        sample_indexes += [x.id for x in tmp]

        # remove all image that contain cls
        ds = ds.match(~F("ground_truth.detections.label").contains(cls))

    zipf_counts: dict = dataset_zipf.count_values("ground_truth.detections.label")
    zipf_counts = dict(sorted(zipf_counts.items(), key=lambda x: x[1], reverse=True))
    print("instance counts: ", zipf_counts)

    plt.figure()
    plt.bar(zipf_counts.keys(), zipf_counts.values())
    plt.savefig("coco_zipf_updated_instances.png")

    image_zipf_count = _get_image_count(dataset_zipf, list(top_classes.keys()))
    image_zipf_count = dict(
        sorted(image_zipf_count.items(), key=lambda x: x[1], reverse=True)
    )
    print("image counts: ", image_zipf_count)
    plt.figure()
    plt.bar(image_zipf_count.keys(), image_zipf_count.values())
    plt.savefig("coco_zipf_updated_images.png")

    return dataset_zipf

def make_coco_2017(
    recreate: bool = typer.Option(False, help="Delete the dataset if it exists")
):
    splits = ["train", "validation"]

    if recreate:
        for s in splits:
            _delete_dataset(DatasetName.coco_2017, split=s)

    for split in splits:
        dataset_name = _kod_dataset_name(DatasetName.coco_2017, split)

        if _dataset_exists(DatasetName.coco_2017, split):
            logging.info(f"Dataset {dataset_name} already exists so skipping it ...")
            continue

        ds: fo.Dataset = foz.load_zoo_dataset("coco-2017", split=split)
        # check if all samples have ground_truth:
        ds = ds.match(F("ground_truth.detections").length() > 0).clone()
        ds.name = dataset_name
        ds.default_classes = [x for x in ds.default_classes if not x.isnumeric()]
        ds.persistent = True
        ds.save()


def make_coco_zipf(
    recreate: bool = typer.Option(False, help="Delete the dataset if it exists"),
    num_classes: int = typer.Option(10, help="Number of classes to keep"),
    zipf_param: float = typer.Option(1.01, help="Zipf parameter"),
    num_samples: int = typer.Option(20000, help="Number of samples to keep"),
):
    dataset_name = _kod_dataset_name(DatasetName.coco_zipf, split="train")

    if recreate:
        _delete_dataset(DatasetName.coco_zipf, split="train")

    if _dataset_exists(DatasetName.coco_zipf, split="train"):
        logging.info(f"Dataset {dataset_name} already exists so skipping it ...")
    else:
        dataset: fo.Dataset = foz.load_zoo_dataset(
            "coco-2017", split="train", max_samples=num_samples * 4
        )

        dataset, top_classes = _get_ds_with_top_classes(dataset, num_classes)

        zipf_distribution = _get_zipf_distribution(num_classes, zipf_param, num_samples)

        dataset_zipf = _convert_dataset_to_zipf(dataset, top_classes, zipf_distribution)

        # save the dataset
        dataset_zipf.name = dataset_name
        dataset_zipf.persistent = True
        dataset_zipf.default_classes = list(top_classes.keys())
        dataset_zipf.save()

    dataset_name = _kod_dataset_name(DatasetName.coco_zipf, split="validation")

    if recreate:
        _delete_dataset(DatasetName.coco_zipf, split="validation")

    if _dataset_exists(DatasetName.coco_zipf, split="validation"):
        logging.info(f"Dataset {dataset_name} already exists so skipping it ...")
        return

    dataset: fo.Dataset = foz.load_zoo_dataset("coco-2017", split="validation")
    dataset = dataset.match(F("ground_truth.detections").length() < 10)
    dataset = dataset.filter_labels(
        "ground_truth", F("label").is_in(list(top_classes.keys()))
    ).clone()

    # saving the dataset
    dataset.name = dataset_name
    dataset.persistent = True
    dataset.default_classes = list(top_classes.keys())
    dataset.save()

    return


def gen_cache(
    dataset_name: DatasetName,
    split: str = typer.Option(..., help="Split"),
):
    ds: fo.Dataset = _load_dataset(dataset_name, split=split)
    ds.compute_metadata(overwrite=True)

    logging.info("Converting to cached format ...")

    cached_samples: list[SampleInfo] = []
    sample: fo.Sample
    for sample in ds.iter_samples(progress=True):
        if sample.metadata is None:
            logging.warning("The sample does not have metadata!")
            continue

        # check if we can load the image
        img_path = sample.filepath
        try:
            img = Image.open(img_path).convert("RGB")
            del img
        except Exception as e:
            logging.warning("Image associated with this sample can not be loaded")
            logging.warning(e)
            continue

        cached_samples.append(_to_cached_sample(sample))

    cache_file = get_default_dataset_cache_dir().joinpath(
        f"kod-{dataset_name.value}-{split}.pkl"
    )

    default_classes = ds.default_classes

    ds_info = DatasetInfo(
        name=f"{dataset_name.value}-{split}",
        date=datetime.now(),
        classes=default_classes,
        samples=cached_samples,
    )

    with open(cache_file, "wb") as fp:
        pickle.dump(ds_info, fp)

    ds_info.summarize()


def do_analysis(dataset_name: DatasetName, output_dir: Path):
    # make sure the the fiftyone dataset exists
    for s in ("train", "validation"):
        if not _dataset_exists(dataset_name, split=s):
            raise ValueError(f"Dataset {dataset_name} does not exist ...")

    # export the datasets to yolo format
    def _export_yolov5(
        train_dataset: fo.Dataset,
        val_dataset: fo.Dataset,
        target_dataset_name: DatasetName,
    ) -> tuple[Path, list[str]]:
        folder_name = str(target_dataset_name.value).replace("-", "_")

        ds_path = get_default_datasets_dir().joinpath("yolov5").joinpath(folder_name)

        print("Exporting yolov5 (train)...")
        train_dataset.export(
            export_dir=str(ds_path),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            split="train",
            classes=train_dataset.default_classes,
        )

        print("Exporting yolov5 (val)...")
        val_dataset.export(
            export_dir=str(ds_path),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            split="val",
            classes=val_dataset.default_classes,
        )

        return ds_path, train_dataset.default_classes

    yv5_ds_path, yv5_ds_classes = _export_yolov5(
        train_dataset=_load_dataset(dataset_name, split="train"),
        val_dataset=_load_dataset(dataset_name, split="validation"),
        target_dataset_name=dataset_name,
    )

    train_loader = YoloFormatDetectionDataset(
        root_dir=yv5_ds_path,
        images_dir="images/train",
        labels_dir="labels/train",
    )

    val_loader = YoloFormatDetectionDataset(
        root_dir=yv5_ds_path,
        images_dir="images/val",
        labels_dir="labels/val",
    )

    analyzer = DetectionAnalysisManager(
        report_title=f"{dataset_name.value} Analysis Report",
        train_data=train_loader,
        val_data=val_loader,
        class_names=yv5_ds_classes,
        log_dir=str(output_dir.joinpath(dataset_name.value)),
        bbox_format="cxcywh",
        is_label_first=True,
    )

    analyzer.run()
