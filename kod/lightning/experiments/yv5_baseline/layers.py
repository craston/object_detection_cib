from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torchvision as tv

from kod.nn.layers.concat import Concat
from kod.core.types import FeatureShape

from .type_defs import PredictionResult


class Yolov5BoxPrediction(nn.Module):
    def __init__(
        self,
        stride: int,
        image_feature_shape: FeatureShape,
        anchor_box_shapes: Sequence[FeatureShape],
    ):
        super().__init__()

        target_feature_shape = FeatureShape(
            width=image_feature_shape.width // stride,
            height=image_feature_shape.height // stride,
        )

        yv, xv = torch.meshgrid(
            [
                torch.arange(target_feature_shape.height).float(),
                torch.arange(target_feature_shape.width).float(),
            ],
            indexing="ij",
        )

        grid = torch.stack((xv, yv), 2).view(
            (
                1,
                1,
                target_feature_shape.height,
                target_feature_shape.width,
                2,
            )
        )

        anchor_boxes_tensor = torch.tensor(anchor_box_shapes)
        anchor_boxes_tensor = anchor_boxes_tensor.reshape(1, -1, 1, 1, 2)

        self.register_buffer("grid", grid)
        self.register_buffer("anchors", anchor_boxes_tensor)

        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # xy
        x[..., 0:2] = x[..., 0:2].sigmoid() * 2 + self.grid.to(x.device) - 0.5
        x[..., 0:2] = x[..., 0:2] * self.stride
        # wh
        x[..., 2:4] = ((x[..., 2:4].sigmoid() * 2) ** 2) * self.anchors.to(x.device)
        x = x.reshape(x.shape[0], -1, 4)
        x = tv.ops.box_convert(x, in_fmt="cxcywh", out_fmt="xyxy")
        return x


class Yolov5ObjectnessPrediction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.sigmoid()
        x = x.reshape(x.shape[0], -1, 1)
        return x


class Yolov5ClassPrediction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.sigmoid()
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x


class Yolov5Prediction(nn.Module):
    def __init__(
        self,
        stride: int,
        image_feature_shape: FeatureShape,
        anchor_box_shapes: Sequence[FeatureShape],
    ):
        super().__init__()

        self.box_prediction = Yolov5BoxPrediction(
            stride,
            image_feature_shape,
            anchor_box_shapes,
        )

        self.obj_prediction = Yolov5ObjectnessPrediction()
        self.class_prediction = Yolov5ClassPrediction()

    def forward(
        self,
        box_features: torch.Tensor,
        obj_features: torch.Tensor,
        class_features: torch.Tensor,
    ) -> PredictionResult:
        box_features = self.box_prediction(box_features)
        obj_features = self.obj_prediction(obj_features)
        class_features = self.class_prediction(class_features)

        return PredictionResult(
            box_features,
            obj_features,
            class_features,
        )


class Yolov5PredictionAssembler(nn.Module):
    def __init__(self):
        super().__init__()

        self.box_concat = Concat(dim=1)
        self.obj_concat = Concat(dim=1)
        self.class_concat = Concat(dim=1)

        self.concat = Concat(dim=-1)

    def forward(
        self,
        box_predictions: Sequence[torch.Tensor],
        obj_predictions: Sequence[torch.Tensor],
        class_predictions: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        all_box_predictions = self.box_concat(box_predictions)
        all_obj_predictions = self.obj_concat(obj_predictions)
        all_class_predictions = self.class_concat(class_predictions)

        all_preds = [
            all_box_predictions,
            all_obj_predictions,
            all_class_predictions,
        ]

        detections = self.concat(all_preds)

        return detections
