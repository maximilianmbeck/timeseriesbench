# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck, Andreas Auer

import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Type, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..base.timeseries_dataset import TimeSeries, TimeSeriesDataset
from ..base.timeseries_traindataset import TimeSeriesTrainDataset
from ..data_mapping import merge_dicts
from ..postprocessing.window_dataset import TimeSeriesWindow

LOGGER = logging.getLogger(__name__)


TARGET_KEY = "y"
FEATURE_KEY = "x"

"""Contains the dataset class for the target dataset. 
This dataset class is a wrapper to a processed time series or time series window.
It takes this timeseries computes the targets and returns the inputs and targets as tensors."""


class TargetGenerator(ABC):
    """Abstract base class for target generators."""

    def get_input_shape(self, timeseries: TimeSeries) -> dict[str, tuple]:
        """Returns the shape of the inputs"""
        input_ = self.get_inputs(timeseries)
        if isinstance(input_, dict):
            return {key: v.shape for key, v in input_.items()}
        else:
            return {FEATURE_KEY: input_.shape}

    def get_inputs(self, timeseries: TimeSeries) -> Union[np.ndarray, dict[str, np.ndarray]]:
        """
        Get all inputs of the feature
        Default behaviour: All features in the time series
        """
        return {
            FEATURE_KEY: timeseries.dataframe[timeseries.dataframe.columns.intersection(timeseries.features)].values
        }

    def get_target_shape(self, timeseries: TimeSeries) -> dict[str, tuple]:
        targets = self.get_targets(timeseries)
        if isinstance(targets, dict):
            return {key: v.shape for key, v in targets.items()}
        else:
            shape = targets.shape if len(targets.shape) > 0 else (1,)
            return {TARGET_KEY: shape}

    @abstractmethod
    def get_targets(self, timeseries: Union[TimeSeries, TimeSeriesWindow]) -> Union[np.ndarray, dict[str, np.ndarray]]:
        pass

    def __call__(self, timeseries: TimeSeries) -> Union[np.ndarray, dict[str, np.ndarray]]:
        return self.get_targets(timeseries)


class NoTarget(TargetGenerator):
    """Target generator that returns no targets.
    This is useful for unsupervised learning."""

    config_class = None

    def get_targets(self, timeseries: Union[TimeSeries, TimeSeriesWindow]) -> Union[np.ndarray, dict[str, np.ndarray]]:
        return {}


def torch_adapter(values: np.ndarray) -> torch.Tensor:
    """Adapter that converts values to torch tensors."""
    values_t = torch.from_numpy(values)
    if values_t.dtype.is_floating_point:
        values_t = values_t.float()
    return values_t


def numpy_adapter(values: np.ndarray) -> np.ndarray:
    """Adapter that converts values to  numpy arrays."""
    return values


class TimeSeriesTargetDataset(TimeSeriesTrainDataset):
    """Time series dataset wrapper that computes the targets for the given time series.
    Returns the inputs and targets as in a dictionary.
    Optionally, caches the items in memory.
    """

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        target_generator: TargetGenerator = NoTarget(),
        output_adapter: Callable[[tuple[np.ndarray, np.ndarray]], tuple] = torch_adapter,
        cache_items: bool = False,
    ):
        assert isinstance(dataset, TimeSeriesDataset), "dataset must be a TimeSeriesDataset"
        assert isinstance(target_generator, TargetGenerator), "target_generator must be a TargetGenerator"
        assert len(dataset) > 0, "dataset must not be empty"

        self.dataset = dataset
        self.target_generator = target_generator
        self.cache_items = cache_items
        self._cache_filled = False

        # shape of the inputs (without batch dimension)
        # (context_length, input_dim)
        self._input_shape = self.target_generator.get_input_shape(self.dataset[0])
        self._output_shape = self.target_generator.get_target_shape(self.dataset[0])
        assert len(self._input_shape[FEATURE_KEY]) == 2, "input shape must be 2 dimensional"
        self._map_output = merge_dicts(value_map_func=output_adapter)

        # cache items in memory
        if self.cache_items:
            self._cached_items = np.empty(len(self.dataset), dtype=object)
            LOGGER.info("Caching items in memory.")
            # fill cache
            for i in tqdm(
                range(len(self.dataset)), desc="Fill Processed Items Cache: ", file=sys.stdout, total=len(self.dataset)
            ):
                self._cached_items[i] = self[i]
            self._cache_filled = True
        else:
            self._cache_items = None

    @property
    def target_dim(self) -> int:
        if (target_shape := self._output_shape.get(TARGET_KEY, None)) is not None:
            if len(target_shape) > 1:
                return target_shape[1]
            else:
                return target_shape[0]
        else:
            return -1

    @property
    def input_dim(self) -> int:
        return self._input_shape[FEATURE_KEY][1]

    @property
    def context_length(self) -> int:
        return self._input_shape[FEATURE_KEY][0]

    @property
    def target_length(self) -> int:
        if (target_shape := self._output_shape.get(TARGET_KEY, None)) is not None and len(target_shape) > 1:
            return target_shape[0]
        else:
            return 1

    @property
    def root(self) -> Any:
        return self.dataset.root

    def __getitem__(self, index: int) -> dict[str, Union[torch.Tensor, np.ndarray]]:
        """Returns the input and target for the given index.
        Returns inputs and targets as tensors for supervised learning or just the inputs for unsupervised learning.
        """
        if self.cache_items and self._cache_filled:
            return self._cached_items[index]
        else:
            timeseries = self.dataset[index]
            inputs = self.target_generator.get_inputs(timeseries)
            if not isinstance(inputs, dict):
                assert (
                    inputs.shape == self._input_shape[FEATURE_KEY]
                ), f"inputs (shape: {inputs.shape}) must have the same shape as the input shape of the dataset {self._input_shape}"
                inputs = {FEATURE_KEY: inputs}
            targets = self.target_generator.get_targets(timeseries)
            if not isinstance(targets, dict):
                targets = {TARGET_KEY: targets}
            return self._map_output(inputs, targets)

    def __len__(self):
        return len(self.dataset)

    def get_meta_data_summary(
        self, meta_data_types: tuple[Type] = (int, str, float), indices: list[int] = [], **kwargs
    ) -> pd.DataFrame:
        return self.dataset.get_meta_data_summary(meta_data_types, indices, **kwargs)
