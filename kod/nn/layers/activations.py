from __future__ import annotations

from functools import partial

import torch.nn as nn

SiLUInplace = partial(nn.SiLU, inplace=True)
