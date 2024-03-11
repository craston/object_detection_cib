from __future__ import annotations

from typing import NamedTuple

import numpy as np


class AugmentedSample(NamedTuple):
    image: np.ndarray
    bboxes: np.ndarray
    labels: np.ndarray
