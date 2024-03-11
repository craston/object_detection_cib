from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from .types import AugmentedSample


def box_candidates(
    orig_bboxes: np.ndarray,
    proc_bboxes: np.ndarray,
    wh_threshold: float = 2,
    aspect_ratio_threshold: float = 20,
    area_thr=0.1,
    eps=1e-7,
):  # box1(4,n), box2(4,n)
    # Compute candidate boxes:
    # box1 before augment,
    # box2 after augment,
    # wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = orig_bboxes[2] - orig_bboxes[0], orig_bboxes[3] - orig_bboxes[1]
    w2, h2 = proc_bboxes[2] - proc_bboxes[0], proc_bboxes[3] - proc_bboxes[1]
    # aspect ratio
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    # candidates
    return (
        (w2 > wh_threshold)
        & (h2 > wh_threshold)
        & (w2 * h2 / (w1 * h1 + eps) > area_thr)
        & (ar < aspect_ratio_threshold)
    )


def _pad_bboxes(
    bboxes: np.ndarray,
    padw=0,
    padh=0,
) -> tuple[np.ndarray, np.ndarray]:
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + padw
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + padh

    return bboxes


class MosaicAugmentor(object):
    def __init__(self, target_image_size: int):
        self.target_size = target_image_size

    def __call__(
        self, input_data: Sequence[AugmentedSample]
    ) -> tuple[AugmentedSample, tuple[int, int]]:
        assert (
            len(input_data) == 4
        ), "mosaic input_data must be a list containing 4 images"

        mosaic_border = (-self.target_size // 2, -self.target_size // 2)
        # mosaic center x, y
        yc, xc = (
            int(random.uniform(-x, 2 * self.target_size + x)) for x in mosaic_border
        )

        proc_bboxes = []
        proc_labels = []
        for i, data in enumerate(input_data):
            img = data.image
            h, w, _ = img.shape

            # place img in img4
            if i == 0:  # top left
                # base image with 4 tiles
                img4 = np.full(
                    (self.target_size * 2, self.target_size * 2, img.shape[2]),
                    114,
                    dtype=np.uint8,
                )
                # xmin, ymin, xmax, ymax (large image)
                x1a, y1a, x2a, y2a = (
                    max(xc - w, 0),
                    max(yc - h, 0),
                    xc,
                    yc,
                )
                # xmin, ymin, xmax, ymax (small image)
                x1b, y1b, x2b, y2b = (
                    w - (x2a - x1a),
                    h - (y2a - y1a),
                    w,
                    h,
                )
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = (
                    xc,
                    max(yc - h, 0),
                    min(xc + w, self.target_size * 2),
                    yc,
                )
                x1b, y1b, x2b, y2b = (
                    0,
                    h - (y2a - y1a),
                    min(w, x2a - x1a),
                    h,
                )
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = (
                    max(xc - w, 0),
                    yc,
                    xc,
                    min(self.target_size * 2, yc + h),
                )
                x1b, y1b, x2b, y2b = (
                    w - (x2a - x1a),
                    0,
                    w,
                    min(y2a - y1a, h),
                )
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = (
                    xc,
                    yc,
                    min(xc + w, self.target_size * 2),
                    min(self.target_size * 2, yc + h),
                )
                x1b, y1b, x2b, y2b = (
                    0,
                    0,
                    min(w, x2a - x1a),
                    min(y2a - y1a, h),
                )

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            if len(data.bboxes) > 0:
                bboxes_xyxy = _pad_bboxes(
                    bboxes=data.bboxes.copy(),
                    padw=padw,
                    padh=padh,
                )
            proc_bboxes.append(bboxes_xyxy)
            proc_labels.append(data.labels)

        proc_bboxes_xyxy = np.concatenate(proc_bboxes, axis=0)
        proc_labels_array = np.concatenate(proc_labels, axis=0)

        proc_bboxes_truncated = np.clip(proc_bboxes_xyxy, 0, 2 * self.target_size)
        i = box_candidates(proc_bboxes_xyxy.T, proc_bboxes_truncated.T)

        proc_bboxes_xyxy = proc_bboxes_xyxy[i]
        np.clip(proc_bboxes_xyxy, 0, 2 * self.target_size - 1, out=proc_bboxes_xyxy)
        proc_labels_array = proc_labels_array[i]

        proc_data = AugmentedSample(
            image=img4,
            bboxes=proc_bboxes_xyxy,
            labels=proc_labels_array,
        )

        return proc_data, mosaic_border
