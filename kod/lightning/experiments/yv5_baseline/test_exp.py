from __future__ import annotations


from kod.nn.networks.yolov5 import Yolov5Network

from kod.core.label_assignment.yv5 import (
    Yolov5LabelAssigner,
    AssignmentAnchorInfo,
)
from kod.core.bbox.iou import IoUCalculator, IoUType

from kod.test_utils.detection_sample import get_batch
from kod.test_utils.anchor_boxes import VOC_BOXES_HL, VOC_BOXES_ML, VOC_BOXES_LL


from .type_defs import LayerwiseAnchorInfo
from .exp import DefaultYolov5Experiment
from .loss import Yolov5Loss, Yolov5LossParams


def make_assigner():
    return Yolov5LabelAssigner(
        anchor_info=AssignmentAnchorInfo(
            ll=VOC_BOXES_LL,
            ml=VOC_BOXES_ML,
            hl=VOC_BOXES_HL,
        )
    )


def test_basic():
    net = Yolov5Network(
        num_anchors_per_cell=3,
        num_classes=80,
        widen_factor=0.25,
        deepen_factor=0.33,
    )
    assigner = make_assigner()
    loss = Yolov5Loss(
        assigner,
        hparams=Yolov5LossParams.get_default(),
        iou_calculator=IoUCalculator(iou_type=IoUType.ciou),
    )

    layerwise_anchor_info = LayerwiseAnchorInfo(
        ll=VOC_BOXES_LL,
        ml=VOC_BOXES_ML,
        hl=VOC_BOXES_HL,
    )

    exp = DefaultYolov5Experiment(
        net=net,
        loss=loss,
        anchor_info=layerwise_anchor_info,
        smart_optimizer=None,
        lr_scheduler=None,
    )

    batch = get_batch()

    exp.training_step(batch, 0)
