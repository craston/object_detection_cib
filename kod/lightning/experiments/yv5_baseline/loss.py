from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv

from kod.nn.networks.yolov5 import Yolov5NetworkResult

from kod.core.types import FeatureShape
from kod.core.bbox.iou import IoUCalculator
from kod.data.detection import DetectionTarget

from kod.core.label_assignment.yv5 import (
    Yolov5LabelAssigner,
    AssignmentTargetInfo,
)

from .type_defs import LossResult


class Yolov5LossParams(NamedTuple):
    lambda_classification: float
    lambda_localization: float
    lambda_objectness: float

    lambda_ll_objectness: float
    lambda_ml_objectness: float
    lambda_hl_objectness: float

    @staticmethod
    def get_default() -> Yolov5LossParams:
        return Yolov5LossParams(
            lambda_classification=0.5,
            lambda_localization=0.05,
            lambda_objectness=1.0,
            lambda_ll_objectness=4.0,
            lambda_ml_objectness=1.0,
            lambda_hl_objectness=0.4,
        )


class Yolov5Loss(nn.Module):
    def __init__(
        self,
        assigner: Yolov5LabelAssigner,
        hparams: Yolov5LossParams,
        iou_calculator: IoUCalculator,
        weights: list[float] = None,
    ):
        super().__init__()
        self.assigner = assigner
        self.hparams = hparams
        self.iou_calculator = iou_calculator
        if weights is not None:
            self.weights = torch.Tensor(weights)
            if torch.cuda.is_available():
                self.weights = self.weights.cuda()
        else:
            self.weights = None

    def _compute_localization_loss(
        self,
        predictions: torch.Tensor,
        target_info: AssignmentTargetInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        filtered_predictions = predictions[
            target_info.indices.samples,
            target_info.indices.anchors,
            target_info.indices.grid_y,
            target_info.indices.grid_x,
        ]

        pred_xy = filtered_predictions[:, :2].sigmoid() * 2 - 0.5
        pred_wh = (
            (filtered_predictions[:, 2:4].sigmoid() * 2) ** 2
        ) * target_info.anchors

        pred_boxes = torch.cat((pred_xy, pred_wh), 1)

        pred_boxes_xyxy = tv.ops.box_convert(
            pred_boxes,
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )
        gt_boxes_xyxy = tv.ops.box_convert(
            target_info.gt_boxes,
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )
        iou = self.iou_calculator(pred_boxes_xyxy, gt_boxes_xyxy).squeeze()

        loss = (1 - iou).mean()

        return loss, iou

    def _compute_objectness_loss(
        self,
        predictions: torch.Tensor,
        target_info: AssignmentTargetInfo,
        iou: torch.Tensor,
        ratio: float,
    ) -> torch.Tensor:
        # iou will also act as the objectness score
        # 1. create an empty tensor
        target_objectness = torch.zeros_like(predictions).squeeze(-1)

        # 2. set the objectness for the filtered predictions
        #    using the iou score
        target_objectness[
            target_info.indices.samples,
            target_info.indices.anchors,
            target_info.indices.grid_y,
            target_info.indices.grid_x,
        ] = iou.clamp(0).type(target_objectness.dtype)

        objectness_loss = F.binary_cross_entropy_with_logits(
            predictions,
            target_objectness.unsqueeze(-1),
            reduction="mean",
        )

        return ratio * objectness_loss

    def _compute_classification_loss(
        self,
        predictions: torch.Tensor,
        target_info: AssignmentTargetInfo,
    ):
        filtered_predictions = predictions[
            target_info.indices.samples,
            target_info.indices.anchors,
            target_info.indices.grid_y,
            target_info.indices.grid_x,
        ]

        # time for classification loss
        #
        # 1. make 1-hot vector
        #    of shape (num_of_filtered_predictions, num_of_classes)
        target_labels_one_hot = torch.zeros_like(filtered_predictions)
        assert target_labels_one_hot.shape[0] == target_info.labels.shape[0]

        # nasty!
        #
        # this is capable of setting 1 at the right location
        target_labels_one_hot[
            range(target_labels_one_hot.shape[0]), target_info.labels
        ] = 1

        # 2. compute the loss
        class_loss = F.binary_cross_entropy_with_logits(
            filtered_predictions,
            target_labels_one_hot,
            reduction="mean",
            pos_weight=self.weights,
        )

        # class_loss = F.cross_entropy(filtered_predictions, target_labels_one_hot)

        return class_loss

    def forward(
        self,
        image_feature_shape: FeatureShape,
        net_result: Yolov5NetworkResult,
        targets: DetectionTarget,
    ) -> LossResult:
        assigned_targets = self.assigner(
            image_feature_shape,
            targets,
        )

        # Localization losses
        ll_loc_loss, ll_iou = self._compute_localization_loss(
            predictions=net_result.ll.box,
            target_info=assigned_targets.ll,
        )
        ml_loc_loss, ml_iou = self._compute_localization_loss(
            predictions=net_result.ml.box,
            target_info=assigned_targets.ml,
        )
        hl_loc_loss, hl_iou = self._compute_localization_loss(
            predictions=net_result.hl.box,
            target_info=assigned_targets.hl,
        )

        # Objectness losses
        ll_obj_loss = self._compute_objectness_loss(
            net_result.ll.obj,
            assigned_targets.ll,
            ll_iou,
            self.hparams.lambda_ll_objectness,
        )
        ml_obj_loss = self._compute_objectness_loss(
            net_result.ml.obj,
            assigned_targets.ml,
            ml_iou,
            self.hparams.lambda_ml_objectness,
        )
        hl_obj_loss = self._compute_objectness_loss(
            net_result.hl.obj,
            assigned_targets.hl,
            hl_iou,
            self.hparams.lambda_hl_objectness,
        )

        # classification losses
        ll_cls_loss = self._compute_classification_loss(
            net_result.ll.cls,
            assigned_targets.ll,
        )
        ml_cls_loss = self._compute_classification_loss(
            net_result.ml.cls,
            assigned_targets.ml,
        )
        hl_cls_loss = self._compute_classification_loss(
            net_result.hl.cls,
            assigned_targets.hl,
        )

        localization_loss = ll_loc_loss + ml_loc_loss + hl_loc_loss
        objectness_loss = ll_obj_loss + ml_obj_loss + hl_obj_loss
        cls_loss = ll_cls_loss + ml_cls_loss + hl_cls_loss

        # In Yolov5 objectness loss and classification loss factors (lambdas)
        # are adjusted based on number of classes and image size
        lambda_objectness = self.hparams.lambda_objectness * (
            (image_feature_shape.width / 640) ** 2
        )

        lambda_classification = self.hparams.lambda_classification * (
            net_result.ll.cls.shape[-1] / 80
        )

        # scale the losses as per the hyperparameters
        scaled_loc_loss = self.hparams.lambda_localization * localization_loss
        scaled_objectness_loss = lambda_objectness * objectness_loss
        scaled_classification_loss = lambda_classification * cls_loss

        return LossResult(
            localization=scaled_loc_loss,
            objectness=scaled_objectness_loss,
            classification=scaled_classification_loss,
        )
