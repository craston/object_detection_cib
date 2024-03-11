from __future__ import annotations

from typing import Sequence
from typing import NamedTuple

from kod.core.types import FeatureShape


class AnchorBoxInfo(NamedTuple):
    stride: int
    boxes_wh: Sequence[FeatureShape]
