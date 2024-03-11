from __future__ import annotations

from typing import Callable
from typing import Sequence

import torch
import torch.nn as nn

from torchvision.ops.misc import Conv2dNormActivation

from kod.nn.layers.activations import SiLUInplace


class SPPFBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: int | Sequence[int] = 5,
        use_conv_first: bool = True,
        mid_channels_scale: float = 0.5,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] = SiLUInplace,
    ):
        super().__init__()

        if use_conv_first:
            mid_channels = int(in_channels * mid_channels_scale)
            self.conv1 = Conv2dNormActivation(
                in_channels,
                mid_channels,
                1,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        else:
            mid_channels = in_channels
            self.conv1 = None

        self.kernel_sizes = kernel_sizes

        self.poolings: nn.MaxPool2d | nn.ModuleList

        if isinstance(kernel_sizes, int):
            self.poolings = nn.MaxPool2d(
                kernel_size=kernel_sizes,
                stride=1,
                padding=kernel_sizes // 2,
            )
            conv2_in_channels = mid_channels * 4
        else:
            self.poolings = nn.ModuleList(
                [
                    nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                    for ks in kernel_sizes
                ]
            )
            conv2_in_channels = mid_channels * (len(kernel_sizes) + 1)

        self.conv2 = Conv2dNormActivation(
            conv2_in_channels,
            out_channels,
            1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv1:
            x = self.conv1(x)

        if isinstance(self.kernel_sizes, int):
            y1 = self.poolings(x)
            y2 = self.poolings(y1)
            x = torch.cat([x, y1, y2, self.poolings(y2)], dim=1)
        else:
            x = torch.cat(
                [x] + [pooling(x) for pooling in self.poolings],  # type: ignore
                dim=1,
            )

        x = self.conv2(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary
    from torchview import draw_graph

    sppf = SPPFBottleneck(256, 256)
    input_size = (1, 256, 20, 20)

    summary(sppf, depth=5, input_size=input_size)

    graph = draw_graph(
        sppf,
        input_size=input_size,
        device="meta",
        expand_nested=True,
        depth=4,
    )

    graph.visual_graph.render("temp/torchview/sppf.dot", format="png")
