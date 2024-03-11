from __future__ import annotations

from kod.data.enums import DatasetName
from kod.data.detection import DetectionSample
from kod.data.detection import DetectionDataset
from kod.data.sample_reader import SampleReader
from kod.data.cache import deserialize_cached_dataset
from kod.data.augmentations.albu import ValidationSampleAugmentor, TrainSampleAugmentor

from kod.lightning.data_module import DetectionDataModule


def get_test_sample(
    image_size: int = 416,
    dataset_name=DatasetName.coco128,
    sample_idx: int = 1,
) -> DetectionSample:
    dataset_info = deserialize_cached_dataset(dataset_name, split="validation")
    sample_reader = SampleReader(
        target_image_size=image_size,
        classes=dataset_info.classes,
        fake_mode=False,
    )

    dataset = DetectionDataset(
        dataset_info,
        sample_reader,
        ValidationSampleAugmentor(),
    )

    return dataset[sample_idx]


def get_batch(
    batch_size: int = 1,
    image_size: int = 416,
    dataset_name=DatasetName.coco128,
):
    dataset_info = deserialize_cached_dataset(dataset_name, split="train")
    sample_reader = SampleReader(
        target_image_size=image_size,
        classes=dataset_info.classes,
        fake_mode=False,
    )

    data_module = DetectionDataModule(
        batch_size,
        sample_reader=sample_reader,
        shuffle=False,
        train_dataset_info=dataset_info,
        train_data_augmentor=TrainSampleAugmentor(augmentations=[]),
    )

    data_module.setup(stage="fit")
    data_loader = data_module.train_dataloader()

    return next(iter(data_loader))
