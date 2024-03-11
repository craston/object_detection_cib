from __future__ import annotations

from typing import NamedTuple

import math

import cv2

import numpy as np

from kod.core.types import FeatureShape
from kod.data.types import AugmentedSample

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .base import SampleAugmentor


class AffineRandValues(NamedTuple):
    perspective_x: float
    perspective_y: float
    degrees: float
    scale: float
    shear_x: float
    shear_y: float
    translate_x: float
    translate_y: float


class AffineParams(NamedTuple):
    degrees: float = 0.0  # image rotation (+/- deg)
    translate: float = 0.1  # image translation (+/- fraction)
    scale: float = 0.5  # image scale (+/- gain)
    shear: float = 0.0  # image shear (+/- deg)
    perspective: float = 0.0  # image perspective (+/- fraction), range 0-0.001

    def should_aug(self) -> bool:
        if (
            self.degrees == 0.0
            and self.translate == 0.0
            and self.scale == 0.0
            and self.shear == 0.0
            and self.perspective == 0.0
        ):
            return False

        return True

    @staticmethod
    def no_aug() -> AffineParams:
        return AffineParams(
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
        )


class HSVParams(NamedTuple):
    hue: float = 0.015  # image HSV-Hue augmentation (fraction)
    saturation: float = 0.7  # image HSV-Saturation augmentation (fraction)
    value: float = 0.4  # image HSV-Value augmentation (fraction)

    def should_aug(self) -> bool:
        if self.hue == 0.0 and self.saturation == 0.0 and self.value == 0.0:
            return False

        return True

    @staticmethod
    def no_aug() -> HSVParams:
        return HSVParams(
            hue=0.0,
            saturation=0.0,
            value=0.0,
        )


class AugParams(NamedTuple):
    affine_params: AffineParams = AffineParams()
    hsv_params: HSVParams = HSVParams()
    flip_lr_prob: float = 0.5
    image_color_transforms: bool = True

    def should_aug(self) -> bool:
        if (
            not self.affine_params.should_aug()
            and not self.hsv_params.should_aug()
            and self.flip_lr_prob == 0.0
            and not self.image_color_transforms
        ):
            return False

        return True

    def should_flip(self, rng: np.random.Generator) -> bool:
        return self.flip_lr_prob > 0.0 and rng.random() < self.flip_lr_prob

    @staticmethod
    def no_aug() -> AugParams:
        return AugParams(
            affine_params=AffineParams.no_aug(),
            hsv_params=HSVParams.no_aug(),
            flip_lr_prob=0.0,
            image_color_transforms=False,
        )


def get_affine_random_values(
    affine_params: AffineParams,
    rng: np.random.Generator,
) -> AffineRandValues:
    perspective_x = rng.uniform(-affine_params.perspective, affine_params.perspective)
    perspective_y = rng.uniform(-affine_params.perspective, affine_params.perspective)

    degree = rng.uniform(-affine_params.degrees, affine_params.degrees)
    scale = rng.uniform(1 - affine_params.scale, 1 + affine_params.scale)

    shear_x = rng.uniform(-affine_params.shear, affine_params.shear)
    shear_y = rng.uniform(-affine_params.shear, affine_params.shear)

    translate_x = rng.uniform(
        0.5 - affine_params.translate, 0.5 + affine_params.translate
    )
    translate_y = rng.uniform(
        0.5 - affine_params.translate, 0.5 + affine_params.translate
    )

    return AffineRandValues(
        perspective_x=perspective_x,
        perspective_y=perspective_y,
        degrees=degree,
        scale=scale,
        shear_x=shear_x,
        shear_y=shear_y,
        translate_x=translate_x,
        translate_y=translate_y,
    )


def _get_feat_shape(width: int, height: int, border=(0, 0)) -> FeatureShape:
    return FeatureShape(width=width + border[1] * 2, height=height + border[0] * 2)


def _get_C(img_shape: FeatureShape, round_off: bool = False) -> np.ndarray:
    C = np.eye(3)
    # x translation (pixels)
    C[0, 2] = -img_shape.width / 2
    # y translation (pixels)
    C[1, 2] = -img_shape.height / 2

    # Round-off chosen only for test purpose to match results with albumentations
    if round_off:
        C = np.round(C)
    return C


def _get_P(perspective_x: float, perspective_y: float) -> np.ndarray:
    P = np.eye(3)
    # x perspective (about y)
    P[2, 0] = perspective_x
    # y perspective (about x)
    P[2, 1] = perspective_y
    return P


def _get_R(degrees: float, scale: float) -> np.ndarray:
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D(angle=degrees, center=(0, 0), scale=scale)
    return R


def _get_S(shear_x: float, shear_y: float) -> np.ndarray:
    S = np.eye(3)
    S[0, 1] = math.tan(shear_x * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(shear_y * math.pi / 180)  # y shear (deg)
    return S


def _get_T(
    translate_x: float, translate_y: float, feat_shape: FeatureShape
) -> np.ndarray:
    T = np.eye(3)
    # x translation (pixels)
    T[0, 2] = translate_x * feat_shape.width
    # y translation (pixels)
    T[1, 2] = translate_y * feat_shape.height
    return T


def _box_candidates(
    orig_bboxes: np.ndarray,
    proc_bboxes: np.ndarray,
    wh_threshold: float = 2,
    aspect_ratio_threshold: float = 20,
    area_thr=0.1,
    eps=1e-16,
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


def _get_combined_matrix(
    affine_values: AffineRandValues,
    feat_shape: FeatureShape,
    feat_shape_with_border: FeatureShape,
    round_off: bool = False,  # Only used for testing
) -> np.ndarray:
    # Center
    C = _get_C(feat_shape, round_off)

    # Perspective
    P = _get_P(affine_values.perspective_x, affine_values.perspective_y)

    # Rotation and Scale
    R = _get_R(affine_values.degrees, affine_values.scale)

    # Shear
    S = _get_S(affine_values.shear_x, affine_values.shear_y)

    # Translation
    T = _get_T(
        affine_values.translate_x,
        affine_values.translate_y,
        feat_shape_with_border,
    )

    # Combined rotation matrix
    # Note - order of operations (right to left) is IMPORTANT
    M = T @ S @ R @ P @ C  # ORDER IS IMPORTANT HERE!!

    return M


def _process_affine_bboxes(
    bboxes: np.ndarray,
    M: np.ndarray,
    feat_shape: FeatureShape,
    perspective: bool,
) -> np.ndarray:
    n = len(bboxes)
    xy = np.ones((n * 4, 3))
    # x1y1, x2y2, x1y2, x2y1
    xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
    xy = xy @ M.T  # transform
    # perspective rescale or affine
    xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)

    # create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    proc_bboxes = (
        np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
    )

    # clip
    # Using width -1 / height -1  to avoid boxes going out of the image
    proc_bboxes[:, [0, 2]] = proc_bboxes[:, [0, 2]].clip(0, feat_shape.width - 1)
    proc_bboxes[:, [1, 3]] = proc_bboxes[:, [1, 3]].clip(0, feat_shape.height - 1)

    return proc_bboxes


def random_perspective(
    input_data: AugmentedSample,
    affine_values: AffineRandValues,
    bbox_wh_threshold: float = 2,
    bbox_aspect_ratio_threshold: float = 20,
    bbox_area_threshold: float = 0.1,
    round_off: bool = False,  # use for testing random_perspective in test_augmentor.py
    border=(0, 0),
) -> AugmentedSample:
    im = input_data.image
    bboxes = input_data.bboxes
    labels = input_data.labels

    feat_shape = _get_feat_shape(width=im.shape[1], height=im.shape[0], border=border)

    # Combined rotation matrix
    M = _get_combined_matrix(
        affine_values=affine_values,
        feat_shape=FeatureShape(width=im.shape[1], height=im.shape[0]),
        feat_shape_with_border=feat_shape,
        round_off=round_off,
    )

    proc_image: np.ndarray = im
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        # image changed
        if affine_values.perspective_x != 0 or affine_values.perspective_y != 0:
            proc_image = cv2.warpPerspective(
                im,
                M,
                dsize=(feat_shape.width, feat_shape.height),
                borderValue=(114, 114, 114),
            )
        else:  # affine
            proc_image = cv2.warpAffine(
                im,
                M[:2],
                dsize=(feat_shape.width, feat_shape.height),
                borderValue=(114, 114, 114),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
    if len(labels) == 0:
        return AugmentedSample(
            image=proc_image,
            bboxes=bboxes,
            labels=labels,
        )

    proc_bboxes = _process_affine_bboxes(
        bboxes=bboxes,
        M=M,
        feat_shape=feat_shape,
        perspective=affine_values.perspective_x != 0.0
        or affine_values.perspective_y != 0.0,
    )

    # filter candidates
    i = _box_candidates(
        orig_bboxes=bboxes.T * affine_values.scale,
        proc_bboxes=proc_bboxes.T,
        wh_threshold=bbox_wh_threshold,
        aspect_ratio_threshold=bbox_aspect_ratio_threshold,
        area_thr=bbox_area_threshold,
    )
    filtered_bboxes = proc_bboxes[i]
    filtered_labels = labels[i]

    return AugmentedSample(
        image=proc_image,
        bboxes=filtered_bboxes,
        labels=filtered_labels,
    )


def augment_hsv(
    img: np.ndarray,
    hsv_params: HSVParams,
    rng: np.random.Generator,
) -> np.ndarray:
    if (
        hsv_params.hue == 0.0
        and hsv_params.saturation == 0.0
        and hsv_params.value == 0.0
    ):
        return img

    r = (
        rng.uniform(-1, 1, 3)
        * [hsv_params.hue, hsv_params.saturation, hsv_params.value]
        + 1
    )
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)

    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def horizontal_flip(input_data: AugmentedSample) -> AugmentedSample:
    image = np.fliplr(input_data.image)
    flipped_bboxes = input_data.bboxes.copy()
    if len(flipped_bboxes) > 0:
        flipped_bboxes[:, 2] = image.shape[1] - 1 - input_data.bboxes[:, 0]
        flipped_bboxes[:, 0] = image.shape[1] - 1 - input_data.bboxes[:, 2]

    return AugmentedSample(
        image=image,
        bboxes=flipped_bboxes,
        labels=input_data.labels,
    )


def mixup(
    input_data1: AugmentedSample, input_data2: AugmentedSample
) -> AugmentedSample:
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = input_data1.image * r + input_data2.image * (1 - r)

    bboxes = np.concatenate((input_data1.bboxes, input_data2.bboxes), 0)
    labels = np.concatenate((input_data1.labels, input_data2.labels), 0)
    return AugmentedSample(im, bboxes, labels)


class TrainSampleAugmentor(SampleAugmentor):
    def __init__(
        self,
        aug_params: AugParams,
        rng_seed: int = 51,
    ):
        self.aug_params = aug_params
        self.rng: np.random.Generator = np.random.default_rng(rng_seed)

        self.image_transform = (
            A.Compose(
                [
                    A.Blur(p=0.01),
                    A.MedianBlur(p=0.01),
                    A.ToGray(p=0.01),
                    A.CLAHE(p=0.01),
                ]
            )
            if aug_params.image_color_transforms
            else None
        )

        self.tensor_transform = A.Compose(
            [
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ]
        )

    def __call__(
        self,
        input_data: AugmentedSample,
        border: tuple[int, int] = (0, 0),
    ) -> AugmentedSample:
        if self.aug_params.affine_params.should_aug():
            affine_values = get_affine_random_values(
                self.aug_params.affine_params,
                rng=self.rng,
            )

            proc_data = random_perspective(
                input_data=input_data,
                affine_values=affine_values,
                border=border,
            )
        else:
            proc_data = input_data

        image = proc_data.image

        if self.image_transform:
            image = self.image_transform(image=image)["image"]

        image = augment_hsv(
            image,
            hsv_params=self.aug_params.hsv_params,
            rng=self.rng,
        )

        # Flip left-right
        hsv_data = AugmentedSample(
            image=image,
            bboxes=proc_data.bboxes,
            labels=proc_data.labels,
        )

        if self.aug_params.should_flip(rng=self.rng):
            flipped_data = horizontal_flip(hsv_data)
        else:
            flipped_data = hsv_data

        aug_img = self.tensor_transform(image=flipped_data.image)["image"]

        return AugmentedSample(
            image=aug_img,
            bboxes=flipped_data.bboxes,
            labels=flipped_data.labels,
        )
