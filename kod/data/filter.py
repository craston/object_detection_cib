from __future__ import annotations

from absl import logging

from .cache import SampleInfo
from .cache import TargetInfo
from .cache import DatasetInfo


def filter_dataset(
    ds_info: DatasetInfo,
    new_name: str,
    classes_to_include: list[str],
) -> DatasetInfo:
    # some sanity checks
    for c in classes_to_include:
        if c not in ds_info.classes:
            raise ValueError(f"{c} is not in the original dataset!")

    filtered_samples: list[SampleInfo] = []
    for s in ds_info.samples:
        filtered_targets: list[TargetInfo] = []
        for t in s.targets:
            if t.class_name in classes_to_include:
                filtered_targets.append(t)

        if filtered_targets:
            filtered_samples.append(
                SampleInfo(
                    id=s.id,
                    image_metadata=s.image_metadata,
                    image_path=s.image_path,
                    targets=filtered_targets,
                )
            )

    logging.info(
        f"Filtering removed {len(ds_info.samples) - len(filtered_samples)} samples..."
    )

    return DatasetInfo(
        name=new_name,
        date=ds_info.date,
        classes=classes_to_include,
        samples=filtered_samples,
    )
