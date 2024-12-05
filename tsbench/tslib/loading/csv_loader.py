# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import pandas as pd

from ..base.timeseries_dataset import TimeSeries, TimeSeriesDataset, TimeSeriesMeta


@dataclass
class CSVTimeSeriesMeta(TimeSeriesMeta):
    csv_meta: dict[str, Any]  # contains the metadata from the csv file (e.g. subject, activity)


class CSVTimeSeries(TimeSeries):
    def __init__(self, index: int, key: str, dataframe: pd.DataFrame, meta_columns: list[str] = []):
        self._raw_dataframe = dataframe
        self._index = index
        self._key = key
        self._meta_columns = meta_columns
        csv_meta = self._get_csv_metadata()
        self._meta_data = CSVTimeSeriesMeta(
            key=key, index=index, csv_meta=csv_meta, features=self.features, num_timesteps=len(self)
        )

    def _get_csv_metadata(self) -> dict[str, Any]:
        """Return one row of the metadata columns as a dictionary."""
        return self._raw_dataframe[self._meta_columns].iloc[0].to_dict()

    @property
    def index(self) -> int:
        return self._index

    @property
    def key(self) -> str:
        return self._key

    @property
    def features(self) -> list[str]:
        return self.dataframe.columns.to_list()  # list(set(self.dataframe.columns.tolist()) - set(self._meta_columns))

    @property
    def dataframe(self) -> pd.DataFrame:
        # exclude the meta columns from the dataframe
        return self._raw_dataframe[self._raw_dataframe.columns.difference(self._meta_columns)]

    def set_dataframe(self, dataframe: pd.DataFrame) -> None:
        self._raw_dataframe = dataframe

    @property
    def meta_data(self) -> CSVTimeSeriesMeta:
        return self._meta_data


def set2str(s: tuple) -> str:
    s_str = [str(i) for i in s]
    return "_".join(s_str)


class CSVTimeSeriesDataset(TimeSeriesDataset):
    def __init__(self, data_file: Union[str, Path], meta_columns: list[str] = []):
        self._meta_columns = meta_columns  # columns that contain metadata (e.g. subject, activity)
        self._full_dataframe = pd.read_csv(data_file)

        self._key_dataframe_dict = self._generate_key_dataframe_dict()
        # generate index to key and key to index maps
        self._index_to_key_map = {}
        self._key_to_index_map = {}
        for i, key in enumerate(self._key_dataframe_dict.keys()):
            self._index_to_key_map[i] = key
            self._key_to_index_map[key] = i

    def _generate_key_dataframe_dict(self) -> dict[str, pd.DataFrame]:
        key_dataframe_dict = {}
        for metadata_set, df in self._full_dataframe.groupby(self._meta_columns):
            key = set2str(metadata_set)
            # for more complicated metadata, you might want to use a more sophisticated key
            # and call it something like self._get_key(df, self._meta_columns)
            key_dataframe_dict[key] = df
        return key_dataframe_dict

    def __getitem__(self, key: Union[str, int]) -> CSVTimeSeries:
        assert isinstance(key, (str, int)), f"Key must be either a string or an integer, got {type(key)}"
        if isinstance(key, int):
            idx = key
            key = self._index_to_key_map[idx]
        else:
            idx = self._key_to_index_map[key]

        ts_dataframe = self._key_dataframe_dict[key]

        return CSVTimeSeries(index=idx, key=key, dataframe=ts_dataframe, meta_columns=self._meta_columns)

    def __len__(self) -> int:
        return len(self._key_dataframe_dict)



class MultiCSVTimeSeriesDataset(TimeSeriesDataset):
    def __init__(self, data_folder: Union[str, Path], meta_columns: list[str] = []):
        self._meta_columns = meta_columns  # columns that contain metadata (e.g. subject, activity)
        self._files = list(Path(data_folder).glob('*.csv'))  # get all csvs in your dir.

        self._file_paths = []

        self._key_dataframe_dict = self._generate_key_dataframe_dict()
        # generate index to key and key to index maps
        self._index_to_key_map = {}
        self._key_to_index_map = {}
        for i, key in enumerate(self._key_dataframe_dict.keys()):
            self._index_to_key_map[i] = key
            self._key_to_index_map[key] = i

    def _generate_key_dataframe_dict(self) -> dict[str, pd.DataFrame]:
        key_dataframe_dict = {}
        for csv_file in self._files:
            file_name = csv_file.stem
            df = pd.read_csv(csv_file)
            if self._meta_columns is not None and len(self._meta_columns) > 0:
                for metadata_set, df in df.groupby(self._meta_columns):
                    key_dataframe_dict[self._get_key(file_name, metadata_set)] = df
            else:
                key_dataframe_dict[self._get_key(file_name, None)] = df
        return key_dataframe_dict

    def _get_key(self, file_name, meta_set):
        if meta_set is not None:
            meta_key = set2str(meta_set)
            return f"{file_name}_{meta_key}"
        else:
            return file_name

    def __getitem__(self, key: Union[str, int]) -> CSVTimeSeries:
        assert isinstance(key, (str, int)), f"Key must be either a string or an integer, got {type(key)}"
        if isinstance(key, int):
            idx = key
            key = self._index_to_key_map[idx]
        else:
            idx = self._key_to_index_map[key]

        ts_dataframe = self._key_dataframe_dict[key]

        return CSVTimeSeries(index=idx, key=key, dataframe=ts_dataframe, meta_columns=self._meta_columns)

    def __len__(self) -> int:
        return len(self._key_dataframe_dict)
