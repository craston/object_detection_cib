from __future__ import annotations

import math

from typing import Iterator
from typing import Optional
from typing import Sequence

import torch
from torch import Generator
from torch.utils.data import Sampler, WeightedRandomSampler

from kod.data.cache import DatasetInfo
from kod.data.filter import filter_dataset


class RandomCycleSampler:
    def __init__(self, data: Sequence[int], generator: Optional[Generator] = None):
        self.data = data
        self.length = len(data)
        self.indices: torch.Tensor = torch.randperm(self.length, generator=generator)
        self.current_index: int = 0
        self.generator = generator

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return self.length

    def __next__(self) -> int:
        if self.current_index == self.length:
            self.indices = torch.randperm(self.length, generator=self.generator)
            self.current_index = 0

        index = self.data[int(self.indices[self.current_index].item())]
        self.current_index += 1
        return index


class ClassAwareSampler(Sampler):

    def __init__(self, dataset_info: DatasetInfo):
        self.dataset_info = dataset_info
        img_ids = [x.id for x in self.dataset_info.samples]
        img_ids_index = dict(zip(img_ids, range(len(img_ids))))

        labels = self.dataset_info.classes
        self.label_to_index = dict(zip(labels, range(len(labels))))

        # List of Classes indices
        self.label_iter_list = RandomCycleSampler(list(self.label_to_index.values()))

        # Per-class image-index list
        self.data_iter_dict: dict[int, Iterator[int]] = dict()
        for k, v in self.label_to_index.items():
            class_dataset = filter_dataset(
                ds_info=self.dataset_info, new_name=k, classes_to_include=[k]
            ).samples
            class_dataset_ids = [x.id for x in class_dataset]
            self.data_iter_dict[v] = RandomCycleSampler(
                [img_ids_index[x] for x in class_dataset_ids]
            )

    def __iter__(self) -> Iterator[int]:
        indices: list[int] = []

        while len(indices) < len(self.dataset_info.samples):
            label_index = next(self.label_iter_list)
            index: int = next(self.data_iter_dict[label_index])
            indices.append(index)

        self.sampler_indices = indices
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset_info.samples)


class RepeatFactorSampler(WeightedRandomSampler):
    def __init__(
        self,
        dataset_info: DatasetInfo,
        reduction: str | None = None,
        threshold: float = 1.0,
        use_sqrt: bool = True,
    ):
        self.dataset_info = dataset_info
        # 1. For each category c, compute the fraction of instances that contain it: f(c)
        class_instance_count = dataset_info.get_instance_count()
        total_instances = sum(class_instance_count.values())
        class_instance_frequency = {
            k: v / total_instances for k, v in class_instance_count.items()
        }

        print("Class Frequency: ", class_instance_frequency)
        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, t / f(c))
        class_repeat_factor = {
            k: max(1.0, (threshold / class_instance_frequency[k]))
            for k in self.dataset_info.classes
        }
        if use_sqrt:
            class_repeat_factor = {
                k: math.sqrt(v) for k, v in class_repeat_factor.items()
            }
        print("Class Instance Repeat Factor: ", class_repeat_factor)
        print(
            "Round Class Instance Repeat Factor: ",
            {k: round(class_repeat_factor[k]) for k in self.dataset_info.classes},
        )
        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = avg_{c in I} r(c)

        self.image_repeat_factors = []

        for sample in self.dataset_info.samples:
            repeat_factor = 0.0
            max_value = 0.0
            for target in sample.targets:
                repeat_factor += class_repeat_factor[target.class_name]
                max_value = max(max_value, class_repeat_factor[target.class_name])

            if reduction == "max":
                self.image_repeat_factors.append(max_value)
            else:
                self.image_repeat_factors.append(
                    repeat_factor / (len(sample.targets) + 1e-6)
                )

        self.generator = torch.Generator()
        self.generator.manual_seed(2023)
        super().__init__(
            torch.tensor(self.image_repeat_factors),
            num_samples=len(self.dataset_info.samples),
            replacement=True,
            generator=self.generator,
        )
