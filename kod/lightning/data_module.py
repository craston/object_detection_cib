from __future__ import annotations

from typing import Callable
from typing import Optional
from typing import Sequence

from absl import logging

import torch
from torch.utils.data import Sampler
from torch.utils.data import DataLoader

import lightning.pytorch as LP

from kod.data.cache import DatasetInfo
from kod.data.cache import SampleInfo
from kod.data.types import AugmentedSample
from kod.data.mosaic import MosaicAugmentor
from kod.data.detection import DetectionDataset
from kod.data.detection import DetectionTarget
from kod.data.detection import DetectionImageInfo


class DetectionDataModule(LP.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        sample_reader: Callable[[SampleInfo, bool], AugmentedSample],
        enable_ram_cache: bool = False,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        train_dataset_info: Optional[DatasetInfo] = None,
        validation_dataset_info: Optional[DatasetInfo] = None,
        prediction_dataset_info: Optional[DatasetInfo] = None,
        train_data_augmentor: Optional[Callable] = None,
        validation_data_augmentor: Optional[Callable] = None,
        prediction_data_augmentor: Optional[Callable] = None,
        mosaic_augmentor: Optional[MosaicAugmentor] = None,
        sampler: Optional[Callable[..., Sampler]] = None,
        collate_fn: Optional[Callable] = None,
        mixup_prob: float = 0.0,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dataset_info = train_dataset_info
        self.validation_dataset_info = validation_dataset_info
        self.prediction_dataset_info = prediction_dataset_info
        self.train_data_augmentor = train_data_augmentor
        self.validation_data_augmentor = validation_data_augmentor
        self.prediction_data_augmentor = prediction_data_augmentor
        self.enable_ram_cache = enable_ram_cache
        self.sample_reader = sample_reader
        self.mosaic_augmentor = mosaic_augmentor
        self.mixup_prob = mixup_prob

        if self.train_dataset_info is not None and sampler is not None:
            self.train_sampler = sampler(self.train_dataset_info)
            # sampler option is mutually exclusive with shuffle
            # for the dataloader
            self.shuffle = False
        else:
            self.train_sampler = None

        self.collate_fn = (
            collate_fn if collate_fn else DetectionDataModule.default_collate_fn
        )

    def prepare_data(self) -> None:
        # As per the doc
        # https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        #
        # Setup is called from the main process.
        # It is not recommended to assign state here as this would
        # not be available for other processes
        pass

    @staticmethod
    def default_collate_fn(
        batch,
    ) -> tuple[torch.Tensor, Sequence[DetectionTarget], Sequence[DetectionImageInfo]]:
        imgs, target, image_info = zip(*batch)
        return torch.stack(imgs, dim=0), target, image_info

    def setup(self, stage: Optional[str] = None) -> None:
        # As per doc
        # https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup

        # These operations are performed on every GPU
        # This is called for every process across all the nodes so state
        # should be set here

        logging.info("Setting up data in datamodule ...")

        if stage == "fit":
            assert self.train_dataset_info is not None
            assert self.train_data_augmentor is not None

            self.train_dataset = DetectionDataset(
                self.train_dataset_info,
                self.sample_reader,
                sample_augmentor=self.train_data_augmentor,
                enable_ram_cache=self.enable_ram_cache,
                mosaic_augmentor=self.mosaic_augmentor,
                mixup_prob=self.mixup_prob,
                sampler=self.train_sampler,
            )

        if stage in ("fit", "validate"):
            if self.validation_dataset_info:
                assert self.validation_data_augmentor is not None
                self.val_dataset = DetectionDataset(
                    self.validation_dataset_info,
                    self.sample_reader,
                    sample_augmentor=self.validation_data_augmentor,
                    enable_ram_cache=self.enable_ram_cache,
                )

        if stage == "predict":
            assert self.prediction_dataset_info is not None
            assert self.prediction_data_augmentor is not None
            self.predict_dataset = DetectionDataset(
                self.prediction_dataset_info,
                self.sample_reader,
                sample_augmentor=self.prediction_data_augmentor,
                enable_ram_cache=self.enable_ram_cache,
            )

    def train_dataloader(self):
        logging.info("Creating train dataloader ...")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collate_fn,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        logging.info("Creating val dataloader ...")
        if not self.validation_dataset_info:
            logging.info("Skipping creation of val dataloader ...")
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        logging.info("Creating prediction dataloader ...")
        if not self.prediction_dataset_info:
            logging.info("Skipping creation of predict dataloader ...")
            return None
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collate_fn,
        )
