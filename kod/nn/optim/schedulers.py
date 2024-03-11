from __future__ import annotations

import math

from functools import partial

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR


def sch_cosine(x: int, max_epochs: int, lrf: float) -> float:
    y1 = 1
    y2 = lrf
    return y1 + 0.5 * (y2 - y1) * (1 - math.cos((x / max_epochs) * math.pi))


def sch_linear(x: int, max_epochs: int, lrf: float) -> float:
    return (1 - x / max_epochs) * (1.0 - lrf) + lrf


def sch_cosine_annealing(x: int, max_epochs: int, lrf: float) -> float:
    return ((1 + math.cos(x * math.pi / max_epochs)) / 2) * (1 - lrf) + lrf


class LinearScheduler(LambdaLR):
    def __init__(
        self,
        lrf: float,
        optimizer: Optimizer,
        max_epochs: int,
        verbose: bool = False,
    ) -> None:
        self.sch_fn = partial(
            sch_linear,
            max_epochs=max_epochs,
            lrf=lrf,
        )

        super().__init__(
            optimizer,
            lr_lambda=self.sch_fn,
            verbose=verbose,
        )


class CosineScheduler(LambdaLR):
    def __init__(
        self,
        lrf: float,
        optimizer: Optimizer,
        max_epochs: int,
        verbose: bool = False,
    ) -> None:
        self.sch_fn = partial(
            sch_linear,
            max_epochs=max_epochs,
            lrf=lrf,
        )

        super().__init__(
            optimizer,
            lr_lambda=partial(
                sch_cosine,
                max_epochs=max_epochs,
                lrf=lrf,
            ),
            verbose=verbose,
        )


class StepScheduler(StepLR):
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int = 100,
        gamma: float = 0.5,
        last_epoch: int = -1,
        max_epochs: int = -1,
        verbose: bool = False,
    ) -> None:
        # this function is used for warmup
        self.sch_fn = partial(
            sch_linear,
            max_epochs=max_epochs,
            lrf=0.1,
        )
        super().__init__(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            verbose=verbose,
        )


class CosineAnnealingScheduler(CosineAnnealingLR):
    def __init__(
        self,
        lrf: float,
        optimizer: Optimizer,
        max_epochs: int,
        verbose: bool = False,
    ) -> None:
        # this function is used for warmup

        self.sch_fn = partial(
            sch_cosine_annealing,
            max_epochs=max_epochs,
            lrf=lrf,
        )
        super().__init__(
            optimizer,
            T_max=max_epochs,
            # eta_min=cfg['lr0'] * lrf
            last_epoch=-1,
        )
