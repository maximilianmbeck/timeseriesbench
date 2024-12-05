# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck


from typing import Optional, Sequence

import torch
from torchmetrics import MetricCollection

from ..tslib.postprocessing.dataset_subset import TimeSeriesTrainDatasetSubset
from ..tslib.target.target_dataset import FEATURE_KEY, TARGET_KEY
from ..tslib.traindataset_generator import (
    TimeSeriesTrainDatasetGenerator,
    TimeSeriesTrainDatasetGeneratorConfig,
)
from .basedataset import BaseSequenceDataset, BaseSequenceDatasetGenerator

"""These are just wrappers making the tslib dataset classes compatible with the tsbench dataset classes."""


class TsLibDataset(BaseSequenceDataset):
    def __init__(self, tstrainds_subset: TimeSeriesTrainDatasetSubset, config: TimeSeriesTrainDatasetGeneratorConfig):
        self._tstrainds_subset = tstrainds_subset
        self.config = config

    def __len__(self) -> int:
        return len(self._tstrainds_subset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # undo the wrapping of the tslib dataset into a dictionary
        # in order to be compatible with the trainer class
        # TODO for later: make the trainer class compatible with the dictionaries of the tslib dataset class

        item = self._tstrainds_subset[index]
        return item[FEATURE_KEY], item[TARGET_KEY]

    @property
    def input_dim(self) -> Sequence[int]:
        return (self._tstrainds_subset.input_dim,)

    @property
    def output_dim(self) -> Sequence[int]:
        return (self._tstrainds_subset.target_dim, self._tstrainds_subset.target_length)

    @property
    def context_length(self) -> int:
        return self._tstrainds_subset.context_length


class TsLibDatasetGenerator(BaseSequenceDatasetGenerator):
    config_class = TimeSeriesTrainDatasetGeneratorConfig

    def __init__(self, config: TimeSeriesTrainDatasetGeneratorConfig):
        self.config = config

        self._tslib_ds_generator = TimeSeriesTrainDatasetGenerator(config)
        self._train_split = None
        self._validation_split = None
        self._test_split = None

    def generate_dataset(self) -> None:
        self._tslib_ds_generator.generate_dataset()

    @property
    def dataset_generated(self) -> bool:
        return self._tslib_ds_generator.dataset_generated

    @property
    def train_split(self) -> TsLibDataset:
        if self._train_split is None:
            self._train_split = TsLibDataset(tstrainds_subset=self._tslib_ds_generator.train_split, config=self.config)
        return self._train_split

    @property
    def validation_split(self) -> TsLibDataset:
        if self._validation_split is None:
            self._validation_split = TsLibDataset(
                tstrainds_subset=self._tslib_ds_generator.validation_split, config=self.config
            )
        return self._validation_split

    @property
    def test_split(self) -> TsLibDataset:
        if self._test_split is None:
            self._test_split = TsLibDataset(tstrainds_subset=self._tslib_ds_generator.test_split, config=self.config)
        return self._test_split

    @property
    def train_metrics(self) -> Optional[MetricCollection]:
        return self._tslib_ds_generator.train_metrics

    @property
    def validation_metrics(self) -> Optional[MetricCollection]:
        return self._tslib_ds_generator.validation_metrics

    @property
    def input_dim(self) -> Sequence[int]:
        return (self._tslib_ds_generator.input_dim,)

    @property
    def output_dim(self) -> Sequence[int]:
        return (self._tslib_ds_generator.target_dim, self._tslib_ds_generator.target_length)

    @property
    def context_length(self) -> int:
        return self._tslib_ds_generator.context_length
