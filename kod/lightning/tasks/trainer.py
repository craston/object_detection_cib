from __future__ import annotations

from typing import Any
from typing import Optional

import torch

import numpy as np

import hydra
import lightning as L
import lightning.pytorch as LP
from lightning.pytorch.loggers import Logger

from omegaconf import DictConfig

from kod.data.enums import DatasetName
from kod.data.cache import deserialize_cached_dataset

from kod.lightning.hydra_utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
)

from kod.lightning.logger import get_logger

from kod.lightning.callbacks.progress import ProgressDisplayCallback
from kod.lightning.callbacks.pycoco_map_eval import PyCOCOMAPEvalCallback

from kod.lightning.hydra_utils.logging import log_hyperparameters
from kod.lightning.hydra_utils.misc import task_wrapper, get_metric_value, extras

log = get_logger(__name__)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    dataset_name = DatasetName(cfg.dataset_name)

    train_dataset_info = deserialize_cached_dataset(dataset_name, split="train")
    val_dataset_info = deserialize_cached_dataset(dataset_name, split="validation")

    log.info(f"Instantiating data module <{cfg.data._target_}>")
    data_module = hydra.utils.instantiate(
        cfg.data,
        sample_reader={"classes": train_dataset_info.classes},
        train_dataset_info=train_dataset_info,
        validation_dataset_info=val_dataset_info,
    )

    if cfg.get("use_loss_weights"):
        dataset_info = train_dataset_info.get_instance_count()
        weights = np.array([v for _, v in dataset_info.items()], dtype=np.float32)
        weights = np.sum(weights) / weights
        log.info(f"Using loss weights: {weights}")
    else:
        weights = None

    log.info(f"Instantiating model <{cfg.model._target_}>")
    exp_module = hydra.utils.instantiate(
        cfg.model,
        net={"num_classes": len(train_dataset_info.classes)},
        loss={"weights": weights},
    )

    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.callbacks)

    if cfg.disable_progress_bar:
        metrics_to_display = exp_module.get_metrics_to_display()
        callbacks.append(
            ProgressDisplayCallback(
                cfg.progress_interval,
                metrics_to_display,
            )
        )

        # remove RichProgressBar if present
        for cb in callbacks:
            if isinstance(cb, LP.callbacks.RichProgressBar):
                cb.disable()

    # add map eval callback
    dataset_info = train_dataset_info.get_instance_count()
    class_names = list(dataset_info.keys())
    # make label info out of it
    label_info = dict(zip(range(len(class_names)), class_names))
    callbacks.append(PyCOCOMAPEvalCallback(label_info))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: LP.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": data_module,
        "model": exp_module,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        exp_module = torch.compile(exp_module)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=exp_module,
            datamodule=data_module,
            ckpt_path=cfg.get("ckpt_path"),
        )

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        # ckpt_path = trainer.checkpoint_callback.best_model_path
        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.validate(model=exp_module, datamodule=data_module, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict,
        metric_name=cfg.get("optimized_metric"),
    )

    # return optimized metric
    return metric_value
