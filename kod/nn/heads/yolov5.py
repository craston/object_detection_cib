from __future__ import annotations

import math
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from kod.nn.heads.types import DetectionHeadResult


class Yolov5BoxHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_anchors_per_cell: int,
    ):
        super().__init__()

        num_predicted_channels = num_anchors_per_cell * 4

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_predicted_channels,
            kernel_size=1,
            stride=1,
        )

        self.pred_reshape = Rearrange(
            "b (num_of_anchors_per_cell p)  h w -> b num_of_anchors_per_cell h w p",
            num_of_anchors_per_cell=num_anchors_per_cell,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv(x)
        x = self.pred_reshape(x)
        return x


class Yolov5ClassificationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_anchors_per_cell: int,
        num_classes: int,
        prior_probability: float = 0.01,
        use_yv5_init: bool = True,
    ):
        super().__init__()

        num_predicted_channels = num_anchors_per_cell * num_classes

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_predicted_channels,
            kernel_size=1,
            stride=1,
        )

        assert conv.bias is not None

        if use_yv5_init:
            update_value = math.log(0.6 / (num_classes - 0.99999))
        else:
            update_value = -math.log((1 - prior_probability) / prior_probability)

        reshaped_conv_bias = conv.bias.view(num_anchors_per_cell, -1)
        new_bias_value = update_value + reshaped_conv_bias.data

        reshaped_conv_bias.data.copy_(new_bias_value)

        self.conv = conv

        self.pred_reshape = Rearrange(
            "b (num_of_anchors_per_cell p)  h w -> b num_of_anchors_per_cell h w p",
            num_of_anchors_per_cell=num_anchors_per_cell,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv(x)
        x = self.pred_reshape(x)
        return x


class Yolov5ObjectnessHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_anchors_per_cell: int,
        stride: int,
        prior_probability: float = 0.01,
        use_yv5_init: bool = True,
    ):
        super().__init__()

        num_predicted_channels = num_anchors_per_cell

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_predicted_channels,
            kernel_size=1,
            stride=1,
        )

        assert conv.bias is not None

        if use_yv5_init:
            update_value = math.log(8 / (640 / stride) ** 2)
        else:
            update_value = -math.log((1 - prior_probability) / prior_probability)

        reshaped_conv_bias = conv.bias.view(num_anchors_per_cell, -1)
        new_bias_value = update_value + reshaped_conv_bias.data

        reshaped_conv_bias.data.copy_(new_bias_value)

        self.conv = conv

        self.pred_reshape = Rearrange(
            "b (num_of_anchors_per_cell p)  h w -> b num_of_anchors_per_cell h w p",
            num_of_anchors_per_cell=num_anchors_per_cell,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv(x)
        x = self.pred_reshape(x)
        return x


class Yolov5Head(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_anchors_per_cell: int,
        num_classes: int,
        stride: int,
        prior_probability: float = 0.01,
        use_yv5_init: bool = True,
    ):
        super().__init__()

        self.box_head = Yolov5BoxHead(
            in_channels=in_channels,
            num_anchors_per_cell=num_anchors_per_cell,
        )
        self.obj_head = Yolov5ObjectnessHead(
            in_channels=in_channels,
            num_anchors_per_cell=num_anchors_per_cell,
            prior_probability=prior_probability,
            stride=stride,
            use_yv5_init=use_yv5_init,
        )
        self.cls_head = Yolov5ClassificationHead(
            in_channels=in_channels,
            num_anchors_per_cell=num_anchors_per_cell,
            num_classes=num_classes,
            prior_probability=prior_probability,
            use_yv5_init=use_yv5_init,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> DetectionHeadResult:
        box = self.box_head(x)
        obj = self.obj_head(x)
        cls = self.cls_head(x)

        return DetectionHeadResult(box, obj, cls)
