# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck, Andreas Auer

from dataclasses import dataclass, field

from torch.utils import data
from torchmetrics import MetricCollection

from ..ml_utils.data.datasetgeneratorinterface import DatasetGeneratorInterface
from ..ml_utils.torch_utils.metrics import get_metric_collection
from .composed_dataset import (
    ComposedTimeSeriesDataset,
    ComposedTimeSeriesDatasetConfig,
    NameAndKwargs,
)
from .postprocessing import create_subset_generator


@dataclass
class TimeSeriesTrainDatasetGeneratorConfig:
    pipeline: ComposedTimeSeriesDatasetConfig
    split: NameAndKwargs
    metric_train: list[NameAndKwargs] = field(default_factory=list)
    metric_val: list[NameAndKwargs] = field(default_factory=list)


class TimeSeriesTrainDatasetGenerator(DatasetGeneratorInterface):
    config_class = TimeSeriesTrainDatasetGeneratorConfig

    def __init__(self, config: TimeSeriesTrainDatasetGeneratorConfig):
        self._config = config
        self._splits = None

    def generate_dataset(self) -> None:
        subset_generator = create_subset_generator(self._config.split)
        self._splits = subset_generator(ComposedTimeSeriesDataset(self._config.pipeline))
        assert len(self._splits) in [2, 3]

    @property
    def dataset_generated(self) -> bool:
        return self._splits is not None

    @property
    def train_split(self) -> data.Dataset:
        return self._splits[0]

    @property
    def validation_split(self) -> data.Dataset:
        return self._splits[1]

    @property
    def test_split(self) -> data.Dataset:
        if len(self._splits) > 2:
            return self._splits[2]
        else:
            raise ValueError("No Test Split available!")

    @property
    def train_metrics(self) -> MetricCollection:
        return get_metric_collection(self._config.metric_train)

    @property
    def validation_metrics(self) -> MetricCollection:
        return get_metric_collection(self._config.metric_val)

    @property
    def target_dim(self) -> int:
        return self._splits[0].target_dim

    @property
    def input_dim(self) -> int:
        return self._splits[0].input_dim

    @property
    def context_length(self) -> int:
        return self._splits[0].context_length

    @property
    def target_length(self) -> int:
        return self._splits[0].target_length

    @property
    def root(self) -> str:
        return self._splits[0].root
