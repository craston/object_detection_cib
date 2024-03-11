from __future__ import annotations

from typing import Optional
from typing import Callable
from typing import NamedTuple

import random
from tqdm import tqdm
from absl import logging

import torch
from torch.utils.data import Dataset as TorchBaseDataset
from torch.utils.data import Sampler

from kod.core.types import FeatureShape

from kod.data.cache import SampleInfo
from kod.data.cache import DatasetInfo
from kod.data.types import AugmentedSample
from kod.data.mosaic import MosaicAugmentor
from kod.data.augmentations.default import mixup


class DetectionTarget(NamedTuple):
    boxes: torch.Tensor
    labels: torch.Tensor


class DetectionImageInfo(NamedTuple):
    image_path: str
    image_shape: FeatureShape


class DetectionSample(NamedTuple):
    img: torch.Tensor
    target: DetectionTarget
    image_info: Optional[DetectionImageInfo] = None


class DetectionDataset(TorchBaseDataset):
    def __init__(
        self,
        dataset_info: DatasetInfo,
        sample_reader: Callable[[SampleInfo, bool], AugmentedSample],
        sample_augmentor: Callable[[AugmentedSample], AugmentedSample],
        enable_ram_cache: bool = False,
        mosaic_augmentor: Optional[MosaicAugmentor] = None,
        mixup_prob: float = 0.0,
        sampler: Sampler = None,
    ):
        self.dataset_info = dataset_info
        self.sample_reader = sample_reader
        self.sample_augmentor = sample_augmentor
        self.enable_ram_cache = enable_ram_cache
        self.mosaic_augmentor = mosaic_augmentor
        self.mixup_prob = mixup_prob
        self.sampler = sampler

        if self.mixup_prob > 0.0:
            assert self.mosaic_augmentor is not None, "Mixup requires mosaic augmentor"

        self._processed_samples: list[AugmentedSample] = [None] * len(
            self.dataset_info.samples
        )

        # Cache the samples
        if enable_ram_cache:
            logging.info("Caching resized images ...")

            for idx, sample in tqdm(
                enumerate(dataset_info.samples),
                total=len(dataset_info.samples),
            ):
                self._processed_samples[idx] = self.sample_reader(
                    sample, self.mosaic_augmentor is None
                )

        self.image_repeat_factors = None
        if hasattr(self.sampler, "image_repeat_factors"):
            self.image_repeat_factors = self.sampler.image_repeat_factors

    def _get_augmented_samples(self, indices: list[int]) -> list[AugmentedSample]:
        augmented_samples: list[AugmentedSample] = []
        for i in indices:
            if self.enable_ram_cache:
                augmented_samples.append(self._processed_samples[i])
            else:
                augmented_samples.append(
                    self.sample_reader(
                        self.dataset_info.samples[i], self.mosaic_augmentor is None
                    )
                )

        return augmented_samples

    def get_num_classes(self) -> int:
        return len(self.dataset_info.classes)

    def __len__(self) -> int:
        return len(self.dataset_info.samples)

    def __getitem__(self, idx: int) -> DetectionSample:
        sample = self.dataset_info.samples[idx]
        image_info = DetectionImageInfo(
            image_path=sample.image_path,
            image_shape=FeatureShape(
                width=sample.image_metadata.width,
                height=sample.image_metadata.height,
            ),
        )

        indices = [idx]

        sampler_indices = range(len(self.dataset_info.samples))
        if hasattr(self.sampler, "sampler_indices"):
            sampler_indices = self.sampler.sampler_indices

        if self.mosaic_augmentor:
            image_info = None
            indices = indices + random.choices(
                sampler_indices, k=3, weights=self.image_repeat_factors
            )
            random.shuffle(indices)

        aug_samples: list[AugmentedSample] = self._get_augmented_samples(indices)

        if self.mosaic_augmentor:
            aug_sample, mosaic_border = self.mosaic_augmentor(aug_samples)
            aug_sample = self.sample_augmentor(aug_sample, mosaic_border)
        else:
            aug_sample = aug_samples[0]
            aug_sample = self.sample_augmentor(aug_sample)

        # Mixup
        if random.random() < self.mixup_prob:
            mixup_indices = random.choices(
                sampler_indices, k=4, weights=self.image_repeat_factors
            )
            mixup_aug_samples: list[AugmentedSample] = self._get_augmented_samples(
                mixup_indices
            )
            mixup_aug_sample, mosaic_border = self.mosaic_augmentor(mixup_aug_samples)
            mixup_aug_sample = self.sample_augmentor(mixup_aug_sample, mosaic_border)

            aug_sample = mixup(aug_sample, mixup_aug_sample)

        target = DetectionTarget(
            boxes=torch.from_numpy(aug_sample.bboxes),
            labels=torch.from_numpy(aug_sample.labels),
        )

        processed_image = aug_sample.image

        return DetectionSample(
            processed_image, target, image_info=image_info  # type: ignore
        )
