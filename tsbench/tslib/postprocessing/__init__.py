# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck, Andreas Auer

from ...ml_utils.meta_factory import get_and_create_class_factory
from .dataset_partitioning import NoFilter
from .dataset_subset import RandomSplitGenerator
from .timeseries_transforms import FeatureSelector, MissingValueHandler, Normalizer

_timeseries_transforms_registry = {
    "feature_selector": FeatureSelector,
    "normalizer": Normalizer,
    "missing_value_handler": MissingValueHandler,
}


_dataset_filter_registry = {"no_filter": NoFilter}

_dataset_subset_generator = {"random_split": RandomSplitGenerator}


get_subset_generator, create_subset_generator = get_and_create_class_factory(_dataset_subset_generator)

get_timeseries_transform, create_timeseries_transform = get_and_create_class_factory(_timeseries_transforms_registry)

get_dataset_filter, create_dataset_filter = get_and_create_class_factory(_dataset_filter_registry)
