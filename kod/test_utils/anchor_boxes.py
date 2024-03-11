from __future__ import annotations

from kod.core.types import FeatureShape
from kod.core.anchors.info import AnchorBoxInfo

VOC_BOXES_HL = AnchorBoxInfo(
    stride=32,
    boxes_wh=[
        FeatureShape(width=116, height=90),
        FeatureShape(width=156, height=198),
        FeatureShape(width=373, height=326),
    ],
)

VOC_BOXES_ML = AnchorBoxInfo(
    stride=16,
    boxes_wh=[
        FeatureShape(width=30, height=61),
        FeatureShape(width=62, height=45),
        FeatureShape(width=59, height=119),
    ],
)

VOC_BOXES_LL = AnchorBoxInfo(
    stride=8,
    boxes_wh=[
        FeatureShape(width=10, height=13),
        FeatureShape(width=16, height=30),
        FeatureShape(width=33, height=23),
    ],
)
