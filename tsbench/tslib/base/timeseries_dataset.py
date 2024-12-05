# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import List, Tuple, Type, Union

import pandas as pd
from tqdm import tqdm

from .base_dataset import BaseDataset

# Architecture Sketch

# data class timeseries meta:
# contains: key, index, features, num_timesteps

# class timeseries (abstract) interface:
# contains: index, key, dataframe, features, duration, get_ts_meta -> rename: property meta_data

# class ts_dataset (abstract) interface:
# contains: get_item -> returns a time series
# > this interface should be inherited by interior dataset


@dataclass
class TimeSeriesMeta:
    key: str
    index: int
    features: List[str]
    num_timesteps: int


class TimeSeries(ABC):
    @property
    @abstractmethod
    def index(self) -> int:
        """Index of the time series in the respective dataset.
        Must not be unique. Depending on the application."""
        pass

    @property
    @abstractmethod
    def key(self) -> str:
        """Unique identifier of the time series."""
        pass

    @property
    @abstractmethod
    def dataframe(self) -> pd.DataFrame:
        """Dataframe containing the time series."""
        pass

    @property
    @abstractmethod
    def meta_data(self) -> TimeSeriesMeta:
        """Meta data of the time series."""
        pass

    @abstractmethod
    def set_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Set the dataframe of the time series."""
        pass

    @property
    def features(self) -> List[str]:
        """List of feature names of the time series."""
        return list(self.dataframe.columns.tolist())

    def __len__(self) -> int:
        return len(self.dataframe)


class TimeSeriesDataset(BaseDataset):
    @abstractmethod
    def __getitem__(self, index: Union[int, str]) -> TimeSeries:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def get_meta_data_summary(
        self, meta_data_types: Tuple[Type] = (int, str, float), indices: List[int] = [], **kwargs
    ) -> pd.DataFrame:
        """Return a dataframe with metadata per sample.

        Args:
            meta_data_types (Tuple[Type], optional): Summarize only these datatypes in the dataframe.
                                                     Defaults to [int, str, float].
            indices (List[int], optional): Summarize only these indices in the dataframe.
                                           Defaults to []. If empty, all indices are summarized.

        Returns:
            pd.DataFrame: Meta data summary
        """
        if len(indices) == 0:
            indices = list(range(len(self)))
        ts_meta = []
        for idx in tqdm(indices, "Sample", file=sys.stdout):  # TODO parallelize
            ts_meta.append({k: v for k, v in asdict(self[idx].meta_data).items() if isinstance(v, meta_data_types)})
        ts_meta_df = pd.DataFrame(ts_meta)
        return ts_meta_df
