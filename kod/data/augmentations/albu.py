from __future__ import annotations

from typing import Sequence
from typing import Protocol
from typing import Optional

import numpy as np


import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import TransformsSeqType

from kod.data.types import AugmentedSample

from .base import SampleAugmentor


class Augmentation(Protocol):
    def get_transform(self) -> TransformsSeqType | None:
        ...


class HorizontalFlipAugmentation(Augmentation):
    def __init__(self, p: float = 0.5):
        self.p = p

    def get_transform(self) -> TransformsSeqType:
        return [
            A.HorizontalFlip(p=self.p),
        ]


class HSVAugmentation(Augmentation):
    def __init__(
        self,
        hue: float = 0.015,
        saturation: float = 0.7,
        value: float = 0.4,
        p: float = 0.5,
    ):
        self.hue = hue
        self.saturation = saturation
        self.value = value
        self.p = p

    def get_transform(self) -> TransformsSeqType:
        return [
            A.ColorJitter(
                brightness=self.value,
                contrast=self.value,
                saturation=self.saturation,
                hue=self.hue,
                p=self.p,
            )
        ]


class BlurAugmentation(Augmentation):
    def __init__(self, p: float = 0.01):
        self.p = p

    def get_transform(self) -> TransformsSeqType:
        return [A.Blur(p=self.p)]


class MedianBlurAugmentation(Augmentation):
    def __init__(self, p: float = 0.01):
        self.p = p

    def get_transform(self) -> TransformsSeqType:
        return [A.MedianBlur(p=self.p)]


class ToGrayAugmentation(Augmentation):
    def __init__(self, p: float = 0.01):
        self.p = p

    def get_transform(self) -> TransformsSeqType:
        return [A.ToGray(p=self.p)]


class CLAHEAugmentation(Augmentation):
    def __init__(self, p: float = 0.01):
        self.p = p

    def get_transform(self) -> TransformsSeqType:
        return [A.CLAHE(p=self.p)]


class ValidationSampleAugmentor(SampleAugmentor):
    def __init__(
        self,
        to_tensor: bool = True,
    ):
        self.tensor_transform = (
            A.Compose(
                [
                    A.ToFloat(max_value=255.0),
                    ToTensorV2(),
                ]
            )
            if to_tensor
            else None
        )

    def __call__(self, sample: AugmentedSample) -> AugmentedSample:
        if self.tensor_transform is not None:
            aug_img = self.tensor_transform(image=sample.image)["image"]
        else:
            aug_img = sample.image

        bboxes = sample.bboxes

        return AugmentedSample(
            bboxes=bboxes,
            labels=sample.labels,
            image=aug_img,
        )


class TrainSampleAugmentor(SampleAugmentor):
    def __init__(
        self,
        augmentations: Optional[Sequence[Augmentation]] = None,
    ):
        all_transforms = []

        if augmentations:
            for a in augmentations:
                all_transforms.extend(a.get_transform())

        # add the to_tensor augmentation
        all_transforms.extend(
            [
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ]
        )

        self.transform = A.Compose(
            all_transforms,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
            ),
        )

    def __call__(self, sample: AugmentedSample) -> AugmentedSample:
        augmented_sample = self.transform(
            image=sample.image,
            bboxes=sample.bboxes.tolist(),
            class_labels=sample.labels.tolist(),
        )

        bboxes = np.array(augmented_sample["bboxes"])

        return AugmentedSample(
            image=augmented_sample["image"],
            bboxes=bboxes,
            labels=np.array(augmented_sample["class_labels"]),
        )
