from __future__ import annotations

from functools import partial

from typing import Callable
from typing import Sequence

import torch
import torch.nn as nn

from torchvision.ops.misc import Conv2dNormActivation

from kod.nn.layers.activations import SiLUInplace


class CSPBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        add_identity: bool = True,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] = SiLUInplace,
    ):
        super().__init__()

        hidden_channels = int(out_channels * expand_ratio)

        self.conv1 = Conv2dNormActivation(
            in_channels,
            hidden_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.conv2 = Conv2dNormActivation(
            hidden_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity

        return out


class CSPBlocks(nn.Sequential):
    def __init__(self, layers: Sequence[nn.Module], **kwargs):
        super().__init__(*layers, **kwargs)


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        add_identity: bool = True,
        num_blocks: int = 1,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] = SiLUInplace,
    ):
        super().__init__()

        mid_channels = int(out_channels * expand_ratio)

        conv_module = partial(
            Conv2dNormActivation,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.short_conv = conv_module(in_channels, mid_channels, kernel_size=1)
        self.main_conv = conv_module(in_channels, mid_channels, kernel_size=1)
        self.last_conv = conv_module(2 * mid_channels, out_channels, kernel_size=1)

        csp_block = partial(
            CSPBlock,
            in_channels=mid_channels,
            out_channels=mid_channels,
            expand_ratio=1.0,  # Note - it is fixed to 1.0 intentionally here
            add_identity=add_identity,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.blocks = CSPBlocks([csp_block() for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat([x_main, x_short], dim=1)

        return self.last_conv(x_final)


if __name__ == "__main__":
    from torchinfo import summary
    from torchview import draw_graph

    csp = CSPLayer(32, 32, add_identity=True, num_blocks=1)
    input_size = (1, 32, 160, 160)

    summary(csp, depth=3, input_size=input_size)

    graph = draw_graph(
        csp,
        input_size=input_size,
        device="meta",
        expand_nested=True,
        depth=3,
    )

    graph.visual_graph.render("temp/torchview/csp_layer.dot", format="png")
