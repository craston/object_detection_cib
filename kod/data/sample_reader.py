from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from PIL import Image

import albumentations as A

from .types import AugmentedSample
from .cache import SampleInfo, TargetInfo


def make_resize_transform(target_image_size: int, letter_boxing: bool = False):
    transforms = [
        A.LongestMaxSize(
            max_size=target_image_size,
            interpolation=cv2.INTER_LINEAR,
        ),
    ]

    if letter_boxing:
        transforms.append(
            A.PadIfNeeded(
                min_height=target_image_size,
                min_width=target_image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
        )

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
        ),
    )


def read_image(
    root_dir: Path,
    sample: SampleInfo,
    fake_mode: bool = False,
) -> np.ndarray:
    if fake_mode:
        return np.random.random(
            size=(
                sample.image_metadata.height,
                sample.image_metadata.width,
                3,
            ),
        )

    image_path = root_dir.joinpath(sample.image_path)
    with Image.open(image_path) as img:
        img = np.array(img.convert("RGB"))
    return img


class SampleReader(object):
    def __init__(
        self,
        target_image_size: int,
        classes: list[str],
        fake_mode: bool = False,
    ):
        self.root_dir = Path.home()
        self.fake_mode = fake_mode
        self.target_image_size = target_image_size
        self.label_to_index = dict(zip(classes, range(len(classes))))

        self.resize_transform = make_resize_transform(
            target_image_size,
            letter_boxing=False,
        )
        self.letter_box_transform = make_resize_transform(
            target_image_size,
            letter_boxing=True,
        )

    def _flatten_targets(
        self,
        targets: list[TargetInfo],
    ) -> tuple[list[list[float]], list[int]]:
        bounding_boxes = []
        class_labels = []

        for target in targets:
            if (target.bounding_box.x_max <= target.bounding_box.x_min) or (
                target.bounding_box.y_max <= target.bounding_box.y_min
            ):
                continue

            bounding_boxes.append(
                [
                    target.bounding_box.x_min,
                    target.bounding_box.y_min,
                    target.bounding_box.x_max,
                    target.bounding_box.y_max,
                ]
            )

            class_labels.append(self.label_to_index[target.class_name])

        return bounding_boxes, class_labels

    def __call__(
        self,
        sample: SampleInfo,
        letter_box: bool = True,
    ) -> AugmentedSample:
        transform = self.letter_box_transform if letter_box else self.resize_transform

        img = read_image(
            self.root_dir,
            sample,
            fake_mode=self.fake_mode,
        )

        # albumentations requires flattened targets
        bboxes, class_labels = self._flatten_targets(sample.targets)

        result = transform(
            image=img,
            bboxes=bboxes,
            class_labels=class_labels,
        )

        img = result["image"]
        bboxes = np.array(result["bboxes"])
        class_labels = np.array(result["class_labels"])

        return AugmentedSample(img, bboxes, class_labels)
