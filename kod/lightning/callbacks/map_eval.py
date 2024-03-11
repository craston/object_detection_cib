from __future__ import annotations

from typing import Sequence
from typing_extensions import TypedDict

import enum

import torch

from kod.data.detection import DetectionTarget


class MAPEvalCallbackInput(TypedDict):
    targets: Sequence[DetectionTarget]
    detections: Sequence[torch.Tensor]


@enum.unique
class MAPLib(enum.Enum):
    pycoco = "pycoco"
