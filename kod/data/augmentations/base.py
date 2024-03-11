from __future__ import annotations

from typing import Protocol

from kod.data.types import AugmentedSample


class SampleAugmentor(Protocol):
    def __call__(self, sample: AugmentedSample) -> AugmentedSample:
        ...
