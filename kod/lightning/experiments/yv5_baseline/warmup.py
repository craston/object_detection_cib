from __future__ import annotations

from typing import Optional
from typing import Callable

import numpy as np

import torch


class OptimizerWarmupUpdater(object):
    def __init__(
        self,
        warmup_epochs: int,
        warmup_bias_lr: float,
        warmup_momentum: float,
        momentum: Optional[float] = None,
    ):
        self.warmup_bias_lr = warmup_bias_lr
        self.warmup_momentum = warmup_momentum
        self.warmup_epochs = warmup_epochs
        self.momentum = momentum

    def __call__(
        self,
        current_step: int,
        current_epoch: int,
        max_warmup_steps: int,
        sch_fn: Callable,
        optimizer: torch.optim.Optimizer,
    ):
        ni = current_step
        nw = max_warmup_steps

        epoch = current_epoch

        # x interp
        xi = [0, nw]
        for pg in optimizer.param_groups:
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            pg["lr"] = np.interp(
                ni,
                xi,
                [
                    self.warmup_bias_lr if pg["name"] == "bias_params" else 0.0,
                    pg["initial_lr"] * sch_fn(epoch),
                ],
            )
            if "momentum" in pg:
                assert self.momentum is not None
                pg["momentum"] = np.interp(
                    ni,
                    xi,
                    [
                        self.warmup_momentum,
                        self.momentum,
                    ],
                )
