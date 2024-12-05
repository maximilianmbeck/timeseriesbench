# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Union

import pandas as pd

from ..base.timeseries_dataset import TimeSeries, TimeSeriesDataset

LOGGER = logging.getLogger(__name__)


class PartitionFilter(ABC):
    @abstractmethod
    def __call__(self, key: str) -> bool:
        pass


class NoFilter(PartitionFilter):
    config_class = None

    def __call__(self, key: str) -> bool:
        return True


class TimeSeriesDatasetPartition(TimeSeriesDataset):
    def __init__(self, dataset: TimeSeriesDataset, partition_filter: PartitionFilter = NoFilter()):
        self.dataset = dataset
        self.partition_filter = partition_filter
        # pindex = partition index
        self._pindex_to_key, self._pindex_to_idx, self._key_to_pindex = self._filter_ts_samples(partition_filter)

    def _filter_ts_samples(
        self, partition_filter: PartitionFilter = NoFilter()
    ) -> Tuple[Dict[int, str], Dict[int, int]]:
        keyslist = self.dataset.keys
        pindex_to_key = {}
        pindex_to_idx = {}
        key_to_pindex = {}

        # this iterates over the dataset keys and filters them by the given filter function
        # the second enumerate is necessary to keep track of the old index
        for partition_idx, (old_index, key) in enumerate(filter(lambda x: partition_filter(x[1]), enumerate(keyslist))):
            pindex_to_key[partition_idx] = key
            pindex_to_idx[partition_idx] = old_index
            key_to_pindex[key] = partition_idx

        return pindex_to_key, pindex_to_idx, key_to_pindex

    def __getitem__(self, index: Union[str, int]) -> TimeSeries:
        if isinstance(index, str):
            index = self._key_to_pindex[index]
        return self.dataset[self._pindex_to_idx[index]]

    def __len__(self):
        return len(self._pindex_to_key)

    @property
    def root(self):
        return self.dataset.root

    def get_meta_data_summary(
        self, meta_data_types: Tuple[Type] = (int, str, float), indices: List[int] = [], **kwargs
    ) -> pd.DataFrame:
        ts_meta_df = self.dataset.get_meta_data_summary(meta_data_types, **kwargs)
        part_ts_meta_df = ts_meta_df.iloc[list(self._pindex_to_idx.values())]
        # keep the old index as extra column:
        # names arg only available in pandas >1.50
        # part_ts_meta_df = part_ts_meta_df.reset_index(names=[f'{self.dataset.__class__.__name__}_index'])
        # legacy version
        part_ts_meta_df = part_ts_meta_df.reset_index().rename(
            columns={"level_0": f"{self.dataset.__class__.__name__}_index"}
        )
        if indices:
            part_ts_meta_df = part_ts_meta_df.iloc[indices]
        return part_ts_meta_df


#! This code is just for reference
# @dataclass
# class VehicleIdFilterConfig:
#     vehicle_ids: List[int] = field(default_factory=list)
#     partition_ids: List[int] = field(default_factory=list)
#     subset_name: str = None


# class VehicleIdFilter(PartitionFilter):
#     config_class = VehicleIdFilterConfig

#     def __init__(self, config: VehicleIdFilterConfig):
#         self._check_config(config)
#         self.config = config
#         self.selected_vehicle_ids = self._get_selected_vehicle_ids(config)
#         LOGGER.info(f"Selected vehicle ids: {self.selected_vehicle_ids}")

#     def _check_config(self, config: VehicleIdFilterConfig):
#         def single_true(iterable):
#             iterator = iter(iterable)
#             # consume from "i" until first true or it's exhausted
#             has_true = any(iterator)
#             # carry on consuming until another true value / exhausted
#             has_another_true = any(iterator)
#             # True if exactly one true found
#             return has_true and not has_another_true

#         assert single_true(
#             [bool(config.vehicle_ids), bool(config.partition_ids), bool(config.subset_name)]
#         ), "Exactly one of the following arguments must be set: vehicle_ids, partition_ids, subset_name"

#     def _get_selected_vehicle_ids(self, config: VehicleIdFilterConfig) -> List[int]:
#         if config.vehicle_ids:
#             return config.vehicle_ids
#         elif config.partition_ids:
#             from ..consts.dataset_consts import VEHICLE_ID_PARTITIONS

#             sel_vehicle_ids = []
#             for partition_id in config.partition_ids:
#                 sel_vehicle_ids += VEHICLE_ID_PARTITIONS[partition_id]
#             return sel_vehicle_ids
#         elif config.subset_name:
#             raise NotImplementedError("TODO: implement subset_name")

#     def __call__(self, key: str) -> bool:
#         return int(key.split("_")[0]) in self.selected_vehicle_ids
