from __future__ import annotations

from kod.core.types import FeatureShape
from kod.test_utils.detection_sample import get_test_sample
from kod.test_utils.anchor_boxes import VOC_BOXES_HL, VOC_BOXES_ML, VOC_BOXES_LL

from .yv5 import Yolov5LabelAssigner, AssignmentAnchorInfo


def test_regular():
    assigner = Yolov5LabelAssigner(
        anchor_info=AssignmentAnchorInfo(
            hl=VOC_BOXES_HL,
            ml=VOC_BOXES_ML,
            ll=VOC_BOXES_LL,
        )
    )

    sample = get_test_sample(sample_idx=1)

    result = assigner(
        input_image_shape=FeatureShape(416, 416),
        targets=[sample.target],
    )

    print(result)
