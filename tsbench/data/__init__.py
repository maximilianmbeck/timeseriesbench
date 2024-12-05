# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from typing import Union

from dacite import Config, from_dict

from ..config import DataConfig
from ..ml_utils.config import NameAndKwargs
from .basedataset import BaseSequenceDatasetGenerator
from .lra.benchmark.listops import ListOpsDatasetGenerator
from .lra.benchmark.sequential_image import (
    Cifar10SequenceDatasetGenerator,
    MNISTSequenceDatasetGenerator,
)
from .tslibwrapper import TsLibDatasetGenerator

_dataset_registry = {
    "smnist": MNISTSequenceDatasetGenerator,
    "scifar10": Cifar10SequenceDatasetGenerator,
    "listops": ListOpsDatasetGenerator,
    "tslib": TsLibDatasetGenerator,
}


def get_datasetgenerator(config: Union[NameAndKwargs, DataConfig]) -> BaseSequenceDatasetGenerator:
    if isinstance(config, (DataConfig, NameAndKwargs)):
        ds_class = _dataset_registry[config.name]
        ds_cfg_class = ds_class.config_class
        ds_cfg = from_dict(data=config.kwargs, data_class=ds_cfg_class, config=Config(strict=True))
        return ds_class(ds_cfg)
    else:
        raise ValueError(f"Unknown config type {type(config)}")
