from __future__ import annotations

from typing import NamedTuple

import torch


class DetectionHeadResult(NamedTuple):
    box: torch.Tensor
    obj: torch.Tensor
    cls: torch.Tensor
