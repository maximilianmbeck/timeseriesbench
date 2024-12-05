# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import logging
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Type

import pandas as pd
from torch.utils.data import Dataset

from ..ml_utils.config import NameAndKwargs
from .base.timeseries_dataset import TimeSeriesDataset
from .base.timeseries_traindataset import T_out, TimeSeriesTrainDataset
from .loading import create_raw_dataset
from .postprocessing import create_dataset_filter, create_timeseries_transform
from .postprocessing.dataset_partitioning import TimeSeriesDatasetPartition
from .postprocessing.timeseries_transforms import (
    FeatureSelector,
    FeatureSelectorConfig,
    Normalizer,
    NormalizerConfig,
    TimeSeriesTransformsDataset,
)
from .postprocessing.window_dataset import (
    TimeSeriesWindowDataset,
    TimeSeriesWindowDatasetConfig,
)
from .target import create_target_generator
from .target.target_dataset import TimeSeriesTargetDataset

LOGGER = logging.getLogger(__name__)


@dataclass
class ComposedTimeSeriesDatasetConfig:
    dataset: NameAndKwargs
    windowing: TimeSeriesWindowDatasetConfig
    partition_filter: NameAndKwargs = field(default_factory=lambda: NameAndKwargs(name="no_filter"))
    feature_selector: FeatureSelectorConfig = field(default_factory=FeatureSelectorConfig)
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)
    transforms: List[NameAndKwargs] = field(default_factory=list)
    target_generator: NameAndKwargs = field(default_factory=lambda: NameAndKwargs(name="no_target"))
    cache_processed_dataset: bool = False


class ComposedTimeSeriesDataset(TimeSeriesTrainDataset, Dataset):
    """Class that contains all dataset wrappers and transformations."""

    def __init__(self, config: ComposedTimeSeriesDatasetConfig):
        self.config = config

        # Datasets are created in the order of the following list:
        self.raw_dataset: TimeSeriesDataset = None
        self.dataset_partition: TimeSeriesDatasetPartition = None
        self.normalized_features_dataset: TimeSeriesTransformsDataset = None
        self.transformed_features_dataset: TimeSeriesTransformsDataset = None
        self.window_dataset: TimeSeriesWindowDataset = None
        self.dataset: TimeSeriesTrainDataset = None
        self._create_composed_dataset()

    def _create_composed_dataset(self):
        LOGGER.info("Generating composed dataset.")
        LOGGER.info("Creating raw dataset.")
        self.raw_dataset = self._create_raw_dataset(dataset_cfg=self.config.dataset)
        LOGGER.info("Creating dataset partition.")
        self.dataset_partition = self._create_dataset_partition(
            dataset=self.raw_dataset, partition_filter_cfg=self.config.partition_filter
        )
        LOGGER.info("Creating normalized features dataset.")
        self.normalized_features_dataset = self._create_normalized_features_dataset(
            dataset=self.dataset_partition,
            feature_selector=self.config.feature_selector,
            normalizer=self.config.normalizer,
        )
        LOGGER.info("Creating transformed features dataset.")
        self.transformed_features_dataset = self._create_transformed_features_dataset(
            dataset=self.normalized_features_dataset, transforms=self.config.transforms
        )
        LOGGER.info("Creating window dataset.")
        self.window_dataset = self._create_window_dataset(
            dataset=self.transformed_features_dataset, window_cfg=self.config.windowing
        )
        LOGGER.info("Creating target generator.")
        self.dataset = self._create_target_dataset(
            dataset=self.window_dataset, target_generator_cfg=self.config.target_generator
        )
        LOGGER.info("Composed dataset created.")

    def _create_raw_dataset(self, dataset_cfg: NameAndKwargs) -> TimeSeriesDataset:
        raw_dataset = create_raw_dataset(dataset_cfg)
        return raw_dataset

    def _create_dataset_partition(self, dataset: TimeSeriesDataset, partition_filter_cfg: NameAndKwargs):
        partition_filter = create_dataset_filter(partition_filter_cfg)
        dataset_partition = TimeSeriesDatasetPartition(dataset=dataset, partition_filter=partition_filter)
        return dataset_partition

    def _create_normalized_features_dataset(
        self, dataset: TimeSeriesDataset, feature_selector: FeatureSelectorConfig, normalizer: NormalizerConfig
    ):
        normalized_features_dataset = TimeSeriesTransformsDataset(
            dataset=dataset,
            transforms=[
                FeatureSelector(feature_selector),
                Normalizer(normalizer),
            ],
        )
        return normalized_features_dataset

    def _create_transformed_features_dataset(
        self, dataset: TimeSeriesDataset, transforms: List[NameAndKwargs]
    ) -> TimeSeriesTransformsDataset:
        timeseries_transforms = [create_timeseries_transform(transform) for transform in transforms]
        transformed_features_dataset = TimeSeriesTransformsDataset(dataset=dataset, transforms=timeseries_transforms)
        return transformed_features_dataset

    def _create_window_dataset(self, dataset: TimeSeriesDataset, window_cfg: TimeSeriesWindowDatasetConfig):
        window_dataset = TimeSeriesWindowDataset(dataset=dataset, config=window_cfg)
        return window_dataset

    def _create_target_dataset(self, dataset: TimeSeriesDataset, target_generator_cfg: NameAndKwargs):
        target_generator = create_target_generator(target_generator_cfg)
        target_dataset = TimeSeriesTargetDataset(
            dataset=dataset, target_generator=target_generator, cache_items=self.config.cache_processed_dataset
        )
        return target_dataset

    @property
    def target_dim(self) -> int:
        return self.dataset.target_dim

    @property
    def input_dim(self) -> int:
        return self.dataset.input_dim

    @property
    def context_length(self) -> int:
        return self.dataset.context_length

    @property
    def target_length(self) -> int:
        return self.dataset.target_length

    @property
    def root(self) -> Any:
        return self.dataset.root

    def __getitem__(self, index: int) -> dict:
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def get_meta_data_summary(
        self, meta_data_types: Tuple[Type] = (int, str, float), indices: List[int] = [], **kwargs
    ) -> pd.DataFrame:
        return self.dataset.get_meta_data_summary(meta_data_types, indices, **kwargs)
