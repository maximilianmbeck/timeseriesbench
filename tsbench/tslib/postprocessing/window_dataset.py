# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import copy
import logging
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..base.timeseries_dataset import TimeSeries, TimeSeriesDataset, TimeSeriesMeta

LOGGER = logging.getLogger(__name__)

# this dataset is a wrapper around a time series dataset
# it splits each time series into multiple windows of a fixed size and indexes these windows

# it returns a window object which contains the time series window as dataframe it also gives access to the original time series, the following time steps and the previous time steps


@dataclass
class WindowIndex:
    start: int
    end: int
    start_past: int
    end_future: int

    @property
    def end_past(self) -> int:
        return self.start

    @property
    def start_future(self) -> int:
        return self.end

    def __str__(self) -> str:
        return f"wi[|{self.start_past}:{self.end_past}|{self.start}:{self.end}|{self.start_future}:{self.end_future}|]"

    def print(self, timeseries=None) -> str:
        if timeseries is None:
            return (
                f"[{self.start_past}:{self.end_past}][{self.start}:{self.end}][{self.start_future}:{self.end_future}]"
            )
        else:
            return f"{timeseries[self.start_past:self.end_past]}{timeseries[self.start:self.end]}{timeseries[self.start_future:self.end_future]}"


def _compute_window_indices_minimal(
    timeseries_length: int, window_size: int, stride: int, initial_offset: int, end_offset: int
) -> Tuple[List[Tuple[int, int]], int, Tuple[int, int]]:
    """
    Computes the start and end indices of the windows of a time series.

    Args:
        timeseries_length (int): The length of the time series.
        window_size (int): Window size in number of time steps.
        stride (int): Offset between windows. Must be positive. If window_size > stride, windows will overlap, no overlap if window_size == stride.
        initial_offset (int): Offset of the first window. Must be non-negative. These time steps will be dropped.
        end_offset (int): Offset of the last window. Must be non-negative. These time steps will be dropped.

    Returns:
        Tuple[List[Tuple[int, int]], int, Tuple[int, int]]: A tuple containing the window indices, the number of dropped steps and the dropped window indices.
    """
    assert timeseries_length > 0, "timeseries_length must be positive"
    assert window_size > 0, "window_size must be positive"
    assert stride > 0, "stride must be positive"
    assert initial_offset >= 0, "initial_offset must be non-negative"
    assert end_offset >= 0, "end_offset must be non-negative"

    # compute the number of windows
    num_windows = int(np.ceil((timeseries_length - window_size - initial_offset - end_offset + 1) / stride))  # + 1

    # compute the window indices
    window_indices = []
    num_dropped_steps = 0
    dropped_window = (0, 0)
    if num_windows > 0:
        for window_idx in range(num_windows):
            window_start_idx = window_idx * stride + initial_offset
            window_end_idx = window_start_idx + window_size
            window_indices.append((window_start_idx, window_end_idx))
        # compute the number of dropped steps
        num_dropped_steps = timeseries_length - window_end_idx - end_offset
        # dropped window
        dropped_window = (window_end_idx, window_end_idx + num_dropped_steps)
    return window_indices, num_dropped_steps, dropped_window


def compute_window_indices(
    timeseries_length: int,
    window_size: int,
    stride: int,
    initial_offset: int,
    end_offset: int,
    future_steps: int = 0,
    past_steps: int = 0,
) -> Tuple[List[WindowIndex], int, Tuple[int, int]]:
    """
    Computes the start and end indices of the windows of a time series.

    Args:
        timeseries_length (int): The length of the time series.
        window_size (int): Window size in number of time steps.
        stride (int): Offset between windows. Must be positive. If window_size > stride, windows will overlap, no overlap if window_size == stride.
        initial_offset (int): Offset of the first window. Must be non-negative. These time steps will be dropped.
        end_offset (int): Offset of the last window. Must be non-negative. These time steps will be dropped.
        future_steps (int, optional): Number of future steps to include in the window. Defaults to 0. If negative, all future steps are included.
        past_steps (int, optional): Number of past steps to include in the window. Defaults to 0. If negative, all past steps are included.

    Returns:
        Tuple[List[WindowIndex], int, Tuple[int, int]]: A tuple containing the window indices, the number of dropped steps and the dropped window indices.
    """
    assert timeseries_length > 0, "timeseries_length must be positive"
    assert window_size > 0, "window_size must be positive"
    assert stride > 0, "stride must be positive"
    assert initial_offset >= 0, "initial_offset must be non-negative"
    assert end_offset >= 0, "end_offset must be non-negative"

    # compute the number of windows
    f_steps = future_steps if future_steps >= 0 else 0
    p_steps = past_steps if past_steps >= 0 else 0
    num_windows = int(
        np.ceil((timeseries_length - window_size - initial_offset - end_offset - f_steps - p_steps + 1) / stride)
    )  # + 1

    # compute the window indices
    window_indices = []
    num_dropped_steps = 0
    dropped_window = (0, 0)
    if num_windows > 0:
        for window_idx in range(num_windows):
            window_start_idx = window_idx * stride + initial_offset + p_steps
            window_end_idx = window_start_idx + window_size
            window_index = WindowIndex(
                start=window_start_idx,
                end=window_end_idx,
                start_past=window_start_idx - p_steps if past_steps >= 0 else 0,
                end_future=window_end_idx + f_steps if future_steps >= 0 else timeseries_length,
            )
            window_indices.append(window_index)
        # compute the number of dropped steps
        num_dropped_steps = timeseries_length - window_end_idx - end_offset
        # dropped window
        dropped_window = (window_end_idx, window_end_idx + num_dropped_steps)
    return window_indices, num_dropped_steps, dropped_window


@dataclass
class TimeSeriesWindowDatasetConfig:
    window_size: int  # number of time steps in a window
    stride: int = 1  # offset between windows, must be positive, if window_size > stride, windows will overlap, no overlap if window_size == stride
    initial_offset: int = 0  # offset of the first window
    end_offset: int = 0  # do not consider the last end_offset time steps
    # -1 means as many as possible
    future_steps: int = 0  # number of future steps to include in the window
    past_steps: int = 0  # number of past steps to include in the window

    # possible extensions:
    # sample windows randomly, restrict to a certain number of windows


class TimeSeriesWindow(TimeSeries):
    """A window of a time series."""

    def __init__(self, timeseries: TimeSeries, window_index: WindowIndex, key: str, ds_index: int):
        self._timeseries = timeseries
        self._window_index = window_index
        # create window meta data from time series meta data
        self._window_meta = copy.deepcopy(timeseries.meta_data)
        self._window_meta.key = key
        self._window_meta.num_timesteps = window_index.end - window_index.start
        self._window_meta.index = ds_index

    @property
    def timeseries(self) -> TimeSeries:
        """The full time series."""
        return self._timeseries

    @property
    def window_index(self) -> WindowIndex:
        """Index of the window in the time series."""
        return self._window_index

    @property
    def index(self) -> int:
        """Index of the window in the timeseries."""
        return self._window_meta.index

    @property
    def key(self) -> str:
        """Key of the window in the dataset."""
        return self._window_meta.key

    @property
    def dataframe(self) -> pd.DataFrame:
        """Dataframe containing the window."""
        return self._timeseries.dataframe.iloc[self._window_index.start : self._window_index.end]

    def set_dataframe(self, dataframe: pd.DataFrame):
        return self._timeseries.set_dataframe(dataframe)

    @property
    def future_dataframe(self) -> pd.DataFrame:
        """Dataframe containing the future time steps of the window."""
        return self._timeseries.dataframe.iloc[self._window_index.start_future : self._window_index.end_future]

    @property
    def past_dataframe(self) -> pd.DataFrame:
        """Dataframe containing the past time steps of the window."""
        return self._timeseries.dataframe.iloc[self._window_index.start_past : self._window_index.end_past]

    @property
    def features(self) -> List[str]:
        return self._timeseries.features

    @property
    def meta_data(self) -> TimeSeriesMeta:
        return self._window_meta


class TimeSeriesWindowDataset(TimeSeriesDataset):
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        config: TimeSeriesWindowDatasetConfig,
    ):
        self.config = config
        self.dataset = dataset
        self._index, self._key_to_index, self._window_meta_data_summary = self._generate_index()

    def _generate_index(self) -> Dict[int, Tuple[int, WindowIndex]]:
        index = {}
        key_to_index = {}
        window_meta_data_summary = []  # create a new metadata summary for the windows

        meta_data_summary = self.dataset.get_meta_data_summary()
        total_dropped_steps = 0
        ds_idx = 0
        # NOTE: we need enumerate(df.iterrows()) to catch the case where self.dataset.get_meta_data_summary()
        # returns a dataset with a non reset index (i.e. an index that does not start with 0)
        for wrapped_ds_idx, (ts_idx, md_row) in tqdm(
            enumerate(meta_data_summary.iterrows()),
            desc="Generating window index",
            file=sys.stdout,
            total=len(meta_data_summary),
        ):
            md_row = md_row.to_dict()
            timeseries_length = md_row["num_timesteps"]
            window_indices, num_dropped_steps, _ = compute_window_indices(
                timeseries_length=timeseries_length,
                window_size=self.config.window_size,
                stride=self.config.stride,
                initial_offset=self.config.initial_offset,
                end_offset=self.config.end_offset,
                future_steps=self.config.future_steps,
                past_steps=self.config.past_steps,
            )
            total_dropped_steps += num_dropped_steps
            for w_idx, window_index in enumerate(window_indices):
                window_key = f"{md_row['key']}_{str(window_index)}"
                index[ds_idx] = (wrapped_ds_idx, window_key, window_index)
                key_to_index[window_key] = ds_idx
                # create a new metadata row for the window
                window_meta_data = copy.deepcopy(md_row)
                window_meta_data["key"] = window_key
                window_meta_data["index"] = ds_idx
                window_meta_data["num_timesteps"] = window_index.end - window_index.start
                window_meta_data_summary.append(window_meta_data)
                ds_idx += 1

        print(f"Total number of dropped timesteps due to windowing: {total_dropped_steps}")
        LOGGER.info(f"Total number of dropped timesteps due to windowing: {total_dropped_steps}")

        window_meta_data_summary = pd.DataFrame(window_meta_data_summary)

        return index, key_to_index, window_meta_data_summary

    def __getitem__(self, idx: Union[int, str]) -> TimeSeriesWindow:
        if isinstance(idx, str):
            idx = self._key_to_index[idx]
        timeseries = self.dataset[self._index[idx][0]]
        window_key = self._index[idx][1]
        window_index = self._index[idx][2]
        return TimeSeriesWindow(
            timeseries=timeseries,
            window_index=window_index,
            key=window_key,
            ds_index=idx,
        )

    def __len__(self) -> int:
        return len(self._index)

    @property
    def root(self) -> str:
        return self.dataset.root

    def get_meta_data_summary(
        self, meta_data_types: Tuple[Type] = (int, str, float), indices: List[int] = [], **kwargs
    ) -> pd.DataFrame:
        if indices:
            return self._window_meta_data_summary.iloc[indices]
        return self._window_meta_data_summary
