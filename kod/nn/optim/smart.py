from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from functools import partial


class SmartOptimizer(object):
    def __init__(
        self,
        optimizer: Callable[..., torch.optim.Optimizer],
        weight_decay: float,
    ):
        self.partial_opt = optimizer
        self.weight_decay = weight_decay

    def __call__(self, net: nn.Module) -> torch.optim.Optimizer:
        # optimizer parameter groups
        bias_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []
        decay_params: list[nn.Parameter] = []
        # normalization layers, e.g BatchNorm2d()
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        for v in net.modules():
            for p_name, p in v.named_parameters(recurse=False):
                if p_name == "bias":  # bias (no decay)
                    bias_params.append(p)
                elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)  # weight (with decay)

        # init with bias params
        optimizer = self.partial_opt(params=bias_params)

        # add name for bias params
        optimizer.param_groups[0]["name"] = "bias_params"

        # add params with weight_decay
        optimizer.add_param_group(
            dict(
                params=decay_params,
                weight_decay=self.weight_decay,
                name="decay_params",
            )
        )

        # add norm params with out weight_decay
        optimizer.add_param_group(
            dict(
                params=no_decay_params,
                weight_decay=0.0,
                name="norm_params",
            )
        )

        return optimizer


if __name__ == "__main__":
    from torch.optim import SGD

    sgd = partial(SGD, lr=0.1, momentum=0.9, nesterov=True)
    smart_opt = SmartOptimizer(sgd, weight_decay=0.0005)

    net = nn.Linear(10, 10)

    opt = smart_opt(net)

    print(opt)
