from __future__ import annotations

from typing import Sequence
from typing import Callable
from typing import NamedTuple

from functools import partial

from absl import logging

import torch
import torch.nn as nn

from kod.nn.backbones.yolov5 import Yolov5Backbone
from kod.nn.backbones.yolov5 import StageConfig
from kod.nn.necks.yolov5_pafpn import Yolov5PAFPN as Yolov5Neck

from kod.nn.heads.yolov5 import Yolov5Head
from kod.nn.heads.types import DetectionHeadResult

from kod.nn.utils import make_divisible
from kod.nn.layers.activations import SiLUInplace

Yolov5BatchNorm2d = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

P5_STAGES = [
    StageConfig(64, 128, 3, True, False),
    StageConfig(128, 256, 6, True, False),
    StageConfig(256, 512, 9, True, False),
    StageConfig(512, 1024, 3, False, True),
]


class Yolov5NetworkResult(NamedTuple):
    ll: DetectionHeadResult
    ml: DetectionHeadResult
    hl: DetectionHeadResult


class Yolov5Network(nn.Module):
    def __init__(
        self,
        num_anchors_per_cell: int,
        num_classes: int,
        norm_layer: Callable[..., nn.Module] = Yolov5BatchNorm2d,
        activation_layer: Callable[..., nn.Module] = SiLUInplace,
        widen_factor: float = 1.0,
        deepen_factor: float = 1.0,
    ):
        super().__init__()

        in_channels_list: Sequence[int] = [
            P5_STAGES[1].out_channels,
            P5_STAGES[2].out_channels,
            P5_STAGES[3].out_channels,
        ]

        self.num_classes = num_classes

        self.backbone = Yolov5Backbone(
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            stages=P5_STAGES,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
        )

        self.neck = Yolov5Neck(
            in_channels_list=in_channels_list,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
        )

        YH = partial(
            Yolov5Head,
            num_anchors_per_cell=num_anchors_per_cell,
            num_classes=num_classes,
        )

        # adjust the channels size
        md = partial(make_divisible, widen_factor=widen_factor)
        in_channels_list = [md(x) for x in in_channels_list]

        self.ll_head = YH(in_channels=in_channels_list[0], stride=8)
        self.ml_head = YH(in_channels=in_channels_list[1], stride=16)
        self.hl_head = YH(in_channels=in_channels_list[2], stride=32)

    def forward(self, x: torch.Tensor) -> Yolov5NetworkResult:
        # get the features from the backbone
        _, ll_features, ml_features, hl_features = self.backbone(x)

        # pass these features to the neck
        ll_features, ml_features, hl_features = self.neck(
            [ll_features, ml_features, hl_features]
        )

        # construct the heads
        ll = self.ll_head(ll_features)
        ml = self.ml_head(ml_features)
        hl = self.hl_head(hl_features)

        return Yolov5NetworkResult(
            ll=ll,
            ml=ml,
            hl=hl,
        )


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)

    import torchinfo
    from torchview import draw_graph

    net = Yolov5Network(
        3,
        80,
        deepen_factor=0.33,
        widen_factor=0.25,
    )
    torchinfo.summary(net, (1, 3, 640, 640))

    graph = draw_graph(
        net,
        input_size=(1, 3, 640, 640),
        device="meta",
        expand_nested=True,
        depth=3,
    )

    graph.visual_graph.render("temp/yolo_v5_n.dot", format="png")
