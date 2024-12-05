# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from ..ml_utils.meta_factory import get_and_create_class_factory
from .dummy_dataset import DummyTsClassificationDsGen
from .traindataset_generator import TimeSeriesTrainDatasetGenerator

_data_generator_registry = {
    "dummy_ts_classification": DummyTsClassificationDsGen,
    "ts_generator": TimeSeriesTrainDatasetGenerator,
}

get_dataset_generator, create_dataset_generator = get_and_create_class_factory(_data_generator_registry)
