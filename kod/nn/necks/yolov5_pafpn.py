from __future__ import annotations

from typing import Sequence
from typing import Callable

import torch
import torch.nn as nn

from torchvision.ops.misc import Conv2dNormActivation

from kod.nn.layers.csp import CSPLayer

from kod.nn.utils import make_divisible, make_round


class Yolov5PAFPN(nn.Module):
    def __init__(
        self,
        in_channels_list: Sequence[int],
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        num_blocks: int = 3,
        expand_ratio: float = 0.5,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
    ):
        super().__init__()

        self.in_channels_list = in_channels_list
        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor
        self.num_blocks = num_blocks

        self.norm_layer = norm_layer
        self.activation_layer = activation_layer

        # Reduce layers
        self.reduce_layers = nn.ModuleList()
        for idx in range(len(in_channels_list)):
            self.reduce_layers.append(self.make_reduction_layer(idx))

        # build top-down blocks
        self.upsample_layers = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()
        for idx in range(len(in_channels_list) - 1, 0, -1):
            self.upsample_layers.append(self.make_upsample_layer())
            self.top_down_layers.append(self.make_top_down_layer(idx))

        # build bottom-up blocks
        self.downsample_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()
        for idx in range(len(in_channels_list) - 1):
            self.downsample_layers.append(self.make_downsample_layer(idx))
            self.bottom_up_layers.append(self.make_bottom_up_layer(idx))

    def make_reduction_layer(
        self,
        idx: int,
    ) -> nn.Module:
        if idx == len(self.in_channels_list) - 1:
            return Conv2dNormActivation(
                in_channels=make_divisible(
                    self.in_channels_list[idx],
                    self.widen_factor,
                ),
                out_channels=make_divisible(
                    self.in_channels_list[idx - 1],
                    self.widen_factor,
                ),
                kernel_size=1,
                norm_layer=self.norm_layer,
                activation_layer=self.activation_layer,
            )

        return nn.Identity()

    def make_top_down_layer(
        self,
        idx: int,
    ) -> nn.Module:
        if idx == 1:
            return CSPLayer(
                in_channels=make_divisible(
                    self.in_channels_list[idx - 1] * 2,
                    self.widen_factor,
                ),
                out_channels=make_divisible(
                    self.in_channels_list[idx - 1],
                    self.widen_factor,
                ),
                num_blocks=make_round(self.num_blocks, self.deepen_factor),
                add_identity=False,
                norm_layer=self.norm_layer,
                activation_layer=self.activation_layer,
            )
        else:
            return nn.Sequential(
                CSPLayer(
                    in_channels=make_divisible(
                        self.in_channels_list[idx - 1] * 2, self.widen_factor
                    ),
                    out_channels=make_divisible(
                        self.in_channels_list[idx - 1], self.widen_factor
                    ),
                    num_blocks=make_round(self.num_blocks, self.deepen_factor),
                    add_identity=False,
                    norm_layer=self.norm_layer,
                    activation_layer=self.activation_layer,
                ),
                Conv2dNormActivation(
                    in_channels=make_divisible(
                        self.in_channels_list[idx - 1],
                        self.widen_factor,
                    ),
                    out_channels=make_divisible(
                        self.in_channels_list[idx - 2],
                        self.widen_factor,
                    ),
                    kernel_size=1,
                    norm_layer=self.norm_layer,
                    activation_layer=self.activation_layer,
                ),
            )

    def make_bottom_up_layer(
        self,
        idx: int,
    ) -> nn.Module:
        return CSPLayer(
            in_channels=make_divisible(
                self.in_channels_list[idx] * 2,
                self.widen_factor,
            ),
            out_channels=make_divisible(
                self.in_channels_list[idx + 1],
                self.widen_factor,
            ),
            num_blocks=make_round(self.num_blocks, self.deepen_factor),
            add_identity=False,
            norm_layer=self.norm_layer,
            activation_layer=self.activation_layer,
        )

    def make_upsample_layer(self) -> nn.Module:
        """build upsample layer."""
        return nn.Upsample(scale_factor=2, mode="nearest")

    def make_downsample_layer(
        self,
        idx: int,
    ) -> nn.Module:
        return Conv2dNormActivation(
            in_channels=make_divisible(
                self.in_channels_list[idx],
                self.widen_factor,
            ),
            out_channels=make_divisible(
                self.in_channels_list[idx],
                self.widen_factor,
            ),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_layer=self.norm_layer,
            activation_layer=self.activation_layer,
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.in_channels_list)

        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels_list)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels_list) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels_list) - 1 - idx](
                feat_high
            )

            top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)

            inner_out = self.top_down_layers[len(self.in_channels_list) - 1 - idx](
                top_down_layer_inputs
            )
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels_list) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        return tuple(outs)
