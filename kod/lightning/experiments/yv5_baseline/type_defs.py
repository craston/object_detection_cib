from __future__ import annotations

from typing import NamedTuple

import torch

from kod.core.anchors.info import AnchorBoxInfo


class PredictionResult(NamedTuple):
    box: torch.Tensor
    obj: torch.Tensor
    cls: torch.Tensor


class LayerwiseAnchorInfo(NamedTuple):
    ll: AnchorBoxInfo
    ml: AnchorBoxInfo
    hl: AnchorBoxInfo


class LayerwisePredictionResult(NamedTuple):
    ll: PredictionResult
    ml: PredictionResult
    hl: PredictionResult


class LossResult(NamedTuple):
    localization: torch.Tensor
    objectness: torch.Tensor
    classification: torch.Tensor
