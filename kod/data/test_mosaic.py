from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import fiftyone as fo


from kod.core.types import FeatureShape

from kod.data.mosaic import MosaicAugmentor
from kod.data.types import AugmentedSample


def visualize(
    orig_data: AugmentedSample,
    proc_data: AugmentedSample,
    left_title: str = "",
    right_title: str = "",
):
    """
    Visualize the original and processed images with bounding boxes and keypoints
    for sanity check.
    """
    ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    ax[0].imshow(orig_data.image)  # original
    ax[1].imshow(proc_data.image)  # processed
    ax[0].set_title(left_title)
    ax[1].set_title(right_title)

    def add_artists(data: AugmentedSample, ax: plt.Axes):
        color = cm.rainbow(np.linspace(0, 1, len(data.bboxes)))
        for i, (box, c) in enumerate(zip(data.bboxes.tolist(), color)):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=c, linewidth=1
            )
            ax.add_patch(rect)

    add_artists(orig_data, ax[0])
    add_artists(proc_data, ax[1])

    plt.show()


def _scale_bboxes(
    bboxes: np.ndarray,
    normalizing_shape: FeatureShape,
    new_shape: FeatureShape,
) -> tuple[np.ndarray, np.ndarray]:
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / (normalizing_shape.width - 1)
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / (normalizing_shape.height - 1)

    bboxes[:, [0, 2]] = (new_shape.width - 1) * bboxes[:, [0, 2]]
    bboxes[:, [1, 3]] = (new_shape.height - 1) * bboxes[:, [1, 3]]

    return bboxes


def resize_sample(
    sample: AugmentedSample, out_size: int, augment: bool = False
) -> AugmentedSample:
    img = sample.image
    orig_h, orig_w, _ = img.shape

    r = out_size / max(orig_h, orig_w)  # ratio
    if r != 1:  # if sizes are not equal
        interp = cv2.INTER_LINEAR if (augment or r > 1) else cv2.INTER_AREA
        img = cv2.resize(
            img,
            (math.ceil(orig_w * r), math.ceil(orig_h * r)),
            interpolation=interp,
        )

    resized_h, resized_w, _ = img.shape

    bboxes_xyxy = _scale_bboxes(
        bboxes=sample.bboxes.copy(),
        normalizing_shape=FeatureShape(width=orig_w, height=orig_h),
        new_shape=FeatureShape(width=resized_w, height=resized_h),
    )

    return AugmentedSample(
        image=img,
        bboxes=bboxes_xyxy,
        labels=sample.labels,
    )


def test_basic() -> None:
    ds: fo.Dataset = fo.load_dataset("kod-voc-toy-validation")

    samples: list[fo.Sample] = ds.take(4)
    sample: fo.Sample
    proc_samples: list[AugmentedSample] = []
    for sample in samples:
        img = np.array(Image.open(sample.filepath).convert("RGB"))
        img_h, img_w, _ = img.shape

        detections: list[fo.Detection] = sample.ground_truth.detections

        bboxes = []
        labels = []

        for detection in detections:
            x1, y1, w, h = detection.bounding_box
            x1 *= img_w
            y1 *= img_h
            x2 = x1 + w * img_w
            y2 = y1 + h * img_h
            bboxes.append([x1, y1, x2, y2])  # x1, y1, x2, y2
            labels.append(detection.label)

        bboxes_array = np.array(bboxes)
        labels_array = np.array(labels)

        original_sample = AugmentedSample(
            img,
            bboxes_array,
            labels_array,
        )

        resized_sample = resize_sample(original_sample, out_size=416, augment=False)
        proc_samples.append(resized_sample)

    mosaic_augmentor = MosaicAugmentor(target_image_size=416)
    mosaic, mosaic_border = mosaic_augmentor(proc_samples)

    visualize(proc_samples[0], mosaic, "one original image", "mosaic")


if __name__ == "__main__":
    test_basic()
