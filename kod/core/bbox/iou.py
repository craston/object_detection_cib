from __future__ import annotations

from typing import NamedTuple

import enum
import torch


@enum.unique
class IoUType(str, enum.Enum):
    ioU = "iou"
    giou = "giou"
    diou = "diou"
    ciou = "ciou"


def fp16_clamp(x: torch, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


class BBCoordinates(NamedTuple):
    x1: torch.Tensor
    y1: torch.Tensor
    x2: torch.Tensor
    y2: torch.Tensor

    def get_width(self) -> torch.Tensor:
        return self.x2 - self.x1

    def get_height(self) -> torch.Tensor:
        return self.y2 - self.y1

    def get_area(self) -> torch.Tensor:
        return self.get_width() * self.get_height()


def _intersection_area(
    boxes1: BBCoordinates,
    boxes2: BBCoordinates,
) -> torch.Tensor:
    x1, y1, x2, y2 = boxes1
    x1g, y1g, x2g, y2g = boxes2

    x1i = torch.max(x1, x1g)
    y1i = torch.max(y1, y1g)
    x2i = torch.min(x2, x2g)
    y2i = torch.min(y2, y2g)

    return (x2i - x1i).clamp(0) * (y2i - y1i).clamp(0)


def _union_area(
    boxes1: BBCoordinates,
    boxes2: BBCoordinates,
    inter: torch.Tensor,
) -> torch.Tensor:
    return boxes1.get_area() + boxes2.get_area() - inter


def _convex_width_height(
    boxes1: BBCoordinates,
    boxes2: BBCoordinates,
) -> tuple[torch.Tensor, torch.Tensor]:
    x1, y1, x2, y2 = boxes1
    x1g, y1g, x2g, y2g = boxes2

    cw = torch.max(x2, x2g) - torch.min(x1, x1g)  # convex width
    ch = torch.max(y2, y2g) - torch.min(y1, y1g)  # convex height

    return cw, ch


def compute_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    assert boxes1.shape == boxes2.shape

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    b1_coords = BBCoordinates(x1, y1, x2, y2)
    b2_coords = BBCoordinates(x1g, y1g, x2g, y2g)

    inter = _intersection_area(b1_coords, b2_coords)
    union = _union_area(b1_coords, b2_coords, inter)

    iou = inter / (union + eps)

    return iou


def compute_iou_unaligned(
    bboxes1: torch.Tensor,
    bboxes2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)

    if rows * cols == 0:
        return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    # [B, rows, cols, 2]
    lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
    # [B, rows, cols, 2]
    rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])

    wh = fp16_clamp(rb - lt, min=0)

    overlap = wh[..., 0] * wh[..., 1]
    union = area1[..., None] + area2[..., None, :] - overlap

    ious = overlap / (union + eps)

    return ious


def compute_giou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    assert boxes1.shape == boxes2.shape

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    b1_coords = BBCoordinates(x1, y1, x2, y2)
    b2_coords = BBCoordinates(x1g, y1g, x2g, y2g)

    inter = _intersection_area(b1_coords, b2_coords)
    union = _union_area(b1_coords, b2_coords, inter)

    iou = inter / (union + eps)

    # compute the penality term
    cw, ch = _convex_width_height(b1_coords, b2_coords)

    convex_area = cw * ch

    penality = torch.abs(convex_area - union) / torch.abs(convex_area + eps)

    return iou - penality


def compute_diou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    assert boxes1.shape == boxes2.shape

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    b1_coords = BBCoordinates(x1, y1, x2, y2)
    b2_coords = BBCoordinates(x1g, y1g, x2g, y2g)

    inter = _intersection_area(b1_coords, b2_coords)
    union = _union_area(b1_coords, b2_coords, inter)

    iou = inter / (union + eps)

    cw, ch = _convex_width_height(b1_coords, b2_coords)

    # convex diagonal squared
    diagonal_distance_squared = cw**2 + ch**2

    # compute center distance squared
    b1_x = (x1 + x2) / 2
    b1_y = (y1 + y2) / 2
    b2_x = (x1g + x2g) / 2
    b2_y = (y1g + y2g) / 2

    centers_distance_squared = (b1_x - b2_x) ** 2 + (b1_y - b2_y) ** 2

    D = centers_distance_squared / (diagonal_distance_squared + eps)

    return iou - D


def compute_ciou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    assert boxes1.shape == boxes2.shape

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    b1_coords = BBCoordinates(x1, y1, x2, y2)
    b2_coords = BBCoordinates(x1g, y1g, x2g, y2g)

    inter = _intersection_area(b1_coords, b2_coords)
    union = _union_area(b1_coords, b2_coords, inter)

    iou = inter / (union + eps)

    cw, ch = _convex_width_height(b1_coords, b2_coords)

    # convex diagonal squared
    diagonal_distance_squared = cw**2 + ch**2

    # compute center distance squared
    b1_x = (x1 + x2) / 2
    b1_y = (y1 + y2) / 2
    b2_x = (x1g + x2g) / 2
    b2_y = (y1g + y2g) / 2

    centers_distance_squared = (b1_x - b2_x) ** 2 + (b1_y - b2_y) ** 2

    D = centers_distance_squared / (diagonal_distance_squared + eps)

    w1, h1 = b1_coords.get_width(), b1_coords.get_height()
    w2, h2 = b2_coords.get_width(), b2_coords.get_height()

    v = (4 / torch.pi**2) * torch.pow(
        torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)),
        2,
    )

    with torch.no_grad():
        alpha = v / ((1 - iou) + v + eps)

    V = alpha * v

    return iou - D - V


class IoUCalculator(object):
    def __init__(self, iou_type: IoUType, eps: float = 1e-7):
        self.iou_type = iou_type
        self.eps = eps

        iou_map: dict[IoUType, callable] = dict(
            iou=compute_iou,
            giou=compute_giou,
            diou=compute_diou,
            ciou=compute_ciou,
        )

        self.fn = iou_map[iou_type]

    def __call__(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
    ):
        return self.fn(boxes1, boxes2, self.eps)
