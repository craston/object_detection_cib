from __future__ import annotations

import enum


@enum.unique
class DatasetName(enum.Enum):
    voc_combined = "voc-combined"
    voc_toy = "voc-toy"
    lvis = "lvis"
    coco128 = "coco128"
    coco_2017 = "coco-2017"
    coco_zipf = "coco-zipf"
    oi_zipf = "oi-zipf"
    coco_zipf_person_sink = "coco-zipf-person-sink"
    coco_zipf_animals = "coco-zipf-animals"
