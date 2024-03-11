from __future__ import annotations

import os
from pathlib import Path


def get_root_dir() -> Path:
    root_dir = os.environ.get("KOD_DATA_ROOT_DIR", None)
    return Path(root_dir) if root_dir else Path.home()


def get_kod_dir() -> Path:
    root_dir = get_root_dir()
    return root_dir.joinpath("kod-data")


def get_default_pretrained_backbone_dir() -> Path:
    hid_od_dir = get_kod_dir()
    model_dir = hid_od_dir.joinpath("pretrained-backbones")
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_default_pretrained_model_dir() -> Path:
    hid_od_dir = get_kod_dir()
    model_dir = hid_od_dir.joinpath("pretrained-models")
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_default_dataset_cache_dir() -> Path:
    hid_od_dir = get_kod_dir()
    cache_dir = hid_od_dir.joinpath("dataset-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_default_datasets_dir() -> Path:
    hid_od_dir = get_kod_dir()
    datasets_dir = hid_od_dir.joinpath("datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    return datasets_dir


def get_default_checkpoint_dir() -> Path:
    hid_od_dir = get_kod_dir()
    checkpoint_dir = hid_od_dir.joinpath("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_default_tensorboard_dir() -> Path:
    hid_od_dir = get_kod_dir()
    tb_dir = hid_od_dir.joinpath("tb_logs")
    tb_dir.mkdir(parents=True, exist_ok=True)
    return tb_dir
