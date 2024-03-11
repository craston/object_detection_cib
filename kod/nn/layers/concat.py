from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.cat(x, self.dim)
