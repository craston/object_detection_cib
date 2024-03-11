from __future__ import annotations

from typing import Sequence
from typing import NamedTuple

import torch


class CXCYWHBoundingBox(NamedTuple):
    cx: float
    cy: float
    width: float
    height: float

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.cx, self.cy, self.width, self.height],
            dtype=torch.float32,
        )

    def scale(
        self,
        x_scale_factor: float,
        y_scale_factor: float,
    ) -> CXCYWHBoundingBox:
        return CXCYWHBoundingBox(
            cx=self.cx * x_scale_factor,
            cy=self.cy * y_scale_factor,
            width=self.width * x_scale_factor,
            height=self.height * y_scale_factor,
        )

    @staticmethod
    def scale_bboxes(
        bboxes: Sequence[CXCYWHBoundingBox],
        x_scale_factor: float,
        y_scale_factor: float,
    ) -> Sequence[CXCYWHBoundingBox]:
        result = []
        for b in bboxes:
            result.append(b.scale(x_scale_factor, y_scale_factor))
        return result

    @staticmethod
    def scale_tensor(
        bboxes: torch.Tensor,
        x_scale_factor: float,
        y_scale_factor: float,
    ) -> torch.Tensor:
        cx = bboxes[..., 0] * x_scale_factor
        cy = bboxes[..., 1] * y_scale_factor
        width = bboxes[..., 2] * x_scale_factor
        height = bboxes[..., 3] * y_scale_factor

        return torch.stack((cx, cy, width, height), dim=-1)

    @staticmethod
    def to_batched_tensor(
        boxes: Sequence[CXCYWHBoundingBox],
    ) -> torch.Tensor:
        tensor_boxes = []
        for b in boxes:
            tensor_boxes.append(b.to_tensor())
        return torch.stack(tensor_boxes, dim=0)


class XYXYBoundingBox(NamedTuple):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def to_cxcywh(self) -> CXCYWHBoundingBox:
        return CXCYWHBoundingBox(
            cx=(self.x_max + self.x_min) / 2,
            cy=(self.y_max + self.y_min) / 2,
            width=self.x_max - self.x_min,
            height=self.y_max - self.y_min,
        )

    @staticmethod
    def from_cxcywh(box: CXCYWHBoundingBox) -> XYXYBoundingBox:
        return XYXYBoundingBox(
            x_min=box.cx - box.width / 2,
            y_min=box.cy - box.height / 2,
            x_max=box.cx + box.width / 2,
            y_max=box.cy + box.height / 2,
        )

    def scale(self, scale_factor: float) -> XYXYBoundingBox:
        return XYXYBoundingBox(
            x_min=self.x_min * scale_factor,
            y_min=self.y_min * scale_factor,
            x_max=self.x_max * scale_factor,
            y_max=self.y_max * scale_factor,
        )

    @staticmethod
    def scale_tensor(bboxes: torch.Tensor, scale_factor: float) -> torch.Tensor:
        x1 = bboxes[..., 0] * scale_factor
        y1 = bboxes[..., 1] * scale_factor
        x2 = bboxes[..., 2] * scale_factor
        y2 = bboxes[..., 3] * scale_factor

        return torch.stack((x1, y1, x2, y2), dim=-1)

    def width(self) -> float:
        return self.x_max - self.x_min

    def height(self) -> float:
        return self.y_max - self.y_min

    def center(self) -> tuple[float, float]:
        return 0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max)

    @staticmethod
    def from_list(box: list[float]) -> XYXYBoundingBox:
        return XYXYBoundingBox(
            x_min=box[0],
            y_min=box[1],
            x_max=box[2],
            y_max=box[3],
        )

    @staticmethod
    def from_list_of_boxes(boxes: list[list[float]]) -> list[XYXYBoundingBox]:
        bboxes: list[XYXYBoundingBox] = []
        for b in boxes:
            bboxes.append(XYXYBoundingBox.from_list(b))
        return bboxes

    def to_list(self) -> list[float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    @staticmethod
    def scale_bboxes(
        bboxes: Sequence[XYXYBoundingBox],
        scale_factor: float,
    ) -> Sequence[XYXYBoundingBox]:
        result = []
        for b in bboxes:
            result.append(b.scale(scale_factor))
        return result

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.x_min, self.y_min, self.x_max, self.y_max],
            dtype=torch.float32,
        )

    @staticmethod
    def to_batched_tensor(
        boxes: Sequence[XYXYBoundingBox],
    ) -> torch.Tensor:
        tensor_boxes = []
        for b in boxes:
            tensor_boxes.append(b.to_tensor())
        return torch.stack(tensor_boxes, dim=0)

    @staticmethod
    def from_tensor(box: torch.Tensor) -> XYXYBoundingBox:
        box_values = box.tolist()
        return XYXYBoundingBox(
            x_min=box_values[0],
            y_min=box_values[1],
            x_max=box_values[2],
            y_max=box_values[3],
        )

    @staticmethod
    def from_batched_tensor(boxes: torch.Tensor) -> Sequence[XYXYBoundingBox]:
        return list(map(XYXYBoundingBox.from_tensor, boxes))
