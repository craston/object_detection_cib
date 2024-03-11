from __future__ import annotations

from typing import Optional
from typing import Callable

from functools import partial

import torch

import lightning.pytorch as LP

from torch.optim.lr_scheduler import LRScheduler

from kod.core.types import FeatureShape
from kod.core.nms import non_max_suppression
from kod.data.detection import DetectionSample
from kod.nn.optim.smart import SmartOptimizer

from kod.nn.networks.yolov5 import (
    Yolov5Network,
    Yolov5NetworkResult,
)

from kod.lightning.callbacks.pycoco_map_eval import MAPEvalCallbackInput

from .loss import Yolov5Loss
from .layers import Yolov5Prediction, Yolov5PredictionAssembler
from .type_defs import (
    LossResult,
    PredictionResult,
    LayerwiseAnchorInfo,
)
from .warmup import OptimizerWarmupUpdater


class DefaultYolov5Experiment(LP.LightningModule):
    def __init__(
        self,
        net: Yolov5Network,
        loss: Yolov5Loss,
        anchor_info: LayerwiseAnchorInfo,
        smart_optimizer: SmartOptimizer,
        lr_scheduler: Callable[..., LRScheduler],
        optimizer_warmup_updater: Optional[OptimizerWarmupUpdater] = None,
        val_nms_conf_threshold: float = 0.001,
        val_nms_iou_threshold: float = 0.6,
    ):
        super().__init__()

        self.net = net
        self.loss = loss
        self.smart_optimizer = smart_optimizer
        self.lr_scheduler = lr_scheduler
        self.anchor_info = anchor_info
        self.optimizer_warmup_updater = optimizer_warmup_updater

        self.val_nms_conf_threshold = val_nms_conf_threshold
        self.val_nms_iou_threshold = val_nms_iou_threshold

    def forward(self, x: torch.Tensor) -> Yolov5NetworkResult | torch.Tensor:
        features: Yolov5NetworkResult = self.net(x)
        if self.training:
            return features
        image_feature_shape = FeatureShape(width=x.shape[-1], height=x.shape[-2])
        return self.get_detections(image_feature_shape, features)

    def get_metrics_to_display(self) -> list[str]:
        return ["box", "cls", "obj"]

    def get_detections(
        self,
        image_feature_shape: FeatureShape,
        net_result: Yolov5NetworkResult,
    ) -> torch.Tensor:
        YP = partial(
            Yolov5Prediction,
            image_feature_shape=image_feature_shape,
        )

        ll_prediction: PredictionResult = YP(
            stride=self.anchor_info.ll.stride,
            anchor_box_shapes=self.anchor_info.ll.boxes_wh,
        )(*net_result.ll)

        ml_prediction: PredictionResult = YP(
            stride=self.anchor_info.ml.stride,
            anchor_box_shapes=self.anchor_info.ml.boxes_wh,
        )(*net_result.ml)

        hl_prediction: PredictionResult = YP(
            stride=self.anchor_info.hl.stride,
            anchor_box_shapes=self.anchor_info.hl.boxes_wh,
        )(*net_result.hl)

        prediction_assembler = Yolov5PredictionAssembler()
        detections = prediction_assembler(
            [ll_prediction.box, ml_prediction.box, hl_prediction.box],
            [ll_prediction.obj, ml_prediction.obj, hl_prediction.obj],
            [ll_prediction.cls, ml_prediction.cls, hl_prediction.cls],
        )

        return detections

    def training_step(
        self,
        batch: DetectionSample,
        batch_idx: int,
    ):
        images, targets, _ = batch
        net_result: Yolov5NetworkResult = self(images)

        image_feature_shape = FeatureShape(
            width=images.shape[3],
            height=images.shape[2],
        )

        # get the predictions/detections
        loss_result: LossResult = self.loss(
            image_feature_shape,
            net_result,
            targets,
        )

        batch_size = images.shape[0]

        total_loss = batch_size * (
            loss_result.localization
            + loss_result.classification
            + loss_result.objectness
        )

        self.log("obj", loss_result.objectness, prog_bar=True, batch_size=batch_size)
        self.log(
            "cls", loss_result.classification, prog_bar=True, batch_size=batch_size
        )
        self.log("box", loss_result.localization, prog_bar=True, batch_size=batch_size)

        return total_loss

    def validation_step(
        self,
        batch: DetectionSample,
        batch_idx: int,
    ):
        images, targets, _ = batch
        detections = self(images)

        filtered_detections = non_max_suppression(
            detections,
            conf_thres=self.val_nms_conf_threshold,
            nms_thres=self.val_nms_iou_threshold,
        )

        return MAPEvalCallbackInput(targets=targets, detections=filtered_detections)

    def configure_optimizers(self):
        optimizer = self.smart_optimizer(self.net)
        self.scheduler = self.lr_scheduler(
            optimizer=optimizer,
            max_epochs=self.trainer.max_epochs,
        )
        return [optimizer], [self.scheduler]

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.optimizer_warmup_updater is None:
            return

        nw = max(
            round(
                self.trainer.num_training_batches
                * self.optimizer_warmup_updater.warmup_epochs
            ),
            100,
        )

        if self.trainer.global_step > nw:
            return

        self.optimizer_warmup_updater(
            current_step=self.trainer.global_step,
            current_epoch=self.trainer.current_epoch,
            max_warmup_steps=nw,
            sch_fn=self.scheduler.sch_fn,
            optimizer=optimizer,
        )
