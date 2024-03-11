from __future__ import annotations

from typing import Sequence
from typing import Callable
from typing import NamedTuple

from functools import partial

import torch
import torch.nn as nn

from torchvision.ops.misc import Conv2dNormActivation

from kod.nn.layers.csp import CSPLayer
from kod.nn.layers.sppf import SPPFBottleneck
from kod.nn.utils import make_divisible, make_round


class StageConfig(NamedTuple):
    in_channels: int
    out_channels: int
    num_blocks: int
    add_identity: bool
    use_spp: bool


class Yolov5Stage(nn.Module):
    def __init__(
        self,
        config: StageConfig,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        deepen_factor: float,
        widen_factor: float,
        expand_ratio: float = 0.5,
        spp_kernel_sizes: int | Sequence[int] = 5,
    ):
        super().__init__()

        in_channels = make_divisible(config.in_channels, widen_factor)
        out_channels = make_divisible(config.out_channels, widen_factor)
        num_blocks = make_round(config.num_blocks, deepen_factor)

        conv = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        components = [conv]

        components.append(
            CSPLayer(
                out_channels,
                out_channels,
                expand_ratio=expand_ratio,
                add_identity=config.add_identity,
                num_blocks=num_blocks,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        if config.use_spp:
            components.append(
                SPPFBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernel_sizes,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.blocks = nn.Sequential(*components)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Yolov5Backbone(nn.Module):
    def __init__(
        self,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        stages: list[StageConfig],
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        spp_kernel_sizes: int | Sequence[int] = 5,
    ):
        super().__init__()

        stem_out_channels = make_divisible(
            stages[0].in_channels,
            widen_factor=widen_factor,
        )

        self.stem = Conv2dNormActivation(
            3,
            stem_out_channels,
            kernel_size=6,
            stride=2,
            padding=2,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        csp_stage = partial(
            Yolov5Stage,
            spp_kernel_sizes=spp_kernel_sizes,
            widen_factor=widen_factor,
            deepen_factor=deepen_factor,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.stages = nn.ModuleDict()
        for idx, config in enumerate(stages):
            self.stages[f"stage{idx+1}"] = csp_stage(config)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x = self.stem(x)
        out_stages = []
        for _, s in self.stages.items():
            x = s(x)
            out_stages.append(x)

        return out_stages
