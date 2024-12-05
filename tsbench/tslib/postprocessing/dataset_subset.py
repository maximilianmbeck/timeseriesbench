# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..base.timeseries_traindataset import T_out, TimeSeriesTrainDataset

"""Contains a dataset class for a subset of a TimeSeriesTrainDataset.
This is used for train, validation and test splits of a dataset.
"""


class TimeSeriesTrainDatasetSubset(TimeSeriesTrainDataset, Dataset):
    def __init__(self, dataset: TimeSeriesTrainDataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = np.array(indices)

    def __getitem__(self, index: int) -> Dict:
        parent_index = self.indices[index]
        return self.dataset[parent_index]

    def __len__(self) -> int:
        return len(self.indices)

    def get_meta_data_summary(
        self, meta_data_types: Tuple[Type] = (int, str, float), indices: List[int] = [], **kwargs
    ) -> pd.DataFrame:
        ts_meta_df = self.dataset.get_meta_data_summary(meta_data_types, **kwargs)
        subset_ts_meta_df = ts_meta_df.iloc[self.indices]
        # keep the old index as extra column:
        # names arg only available in pandas >1.50
        # part_ts_meta_df = part_ts_meta_df.reset_index(names=[f'{self.dataset.__class__.__name__}_index'])
        # legacy version
        subset_ts_meta_df = subset_ts_meta_df.reset_index().rename(
            columns={"level_0": f"{self.dataset.__class__.__name__}_index"}
        )
        if indices:
            subset_ts_meta_df = subset_ts_meta_df.iloc[indices]
        return subset_ts_meta_df

    @property
    def root(self) -> Any:
        return self.dataset.root

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


class SubsetGenerator(ABC):
    @abstractmethod
    def __call__(self, dataset: TimeSeriesTrainDataset) -> List[TimeSeriesTrainDatasetSubset]:
        pass


def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # Taken from python 3.5 docs
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


@dataclass
class RandomSplitConfig:
    lengths: Sequence[Union[int, float]]


class RandomSplitGenerator(SubsetGenerator):
    config_class = RandomSplitConfig

    def __init__(self, config: RandomSplitConfig) -> None:
        self._config = config

    def __call__(self, dataset: TimeSeriesTrainDataset) -> List[TimeSeriesTrainDatasetSubset]:
        return self.random_split(dataset, lengths=self._config.lengths)

    @staticmethod
    def random_split(
        dataset: TimeSeriesTrainDataset,
        lengths: Sequence[Union[int, float]],
        generator: Optional[np.random.Generator] = np.random.default_rng(seed=0),
    ) -> List[TimeSeriesTrainDatasetSubset]:
        r"""
        Adapted from torch.utils.data.random_split.

        Randomly split a dataset into non-overlapping new datasets of given lengths.

        If a list of fractions that sum up to 1 is given,
        the lengths will be computed automatically as
        floor(frac * len(dataset)) for each fraction provided.

        After computing the lengths, if there are any remainders, 1 count will be
        distributed in round-robin fashion to the lengths
        until there are no remainders left.

        Optionally fix the generator for reproducible results, e.g.:

        >>> random_split(range(10), [3, 7], seed=42)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
        ...   ).manual_seed(42))

        Args:
            dataset (Dataset): Dataset to be split
            lengths (sequence): lengths or fractions of splits to be produced
            generator (Generator): Generator used for the random permutation.
        """

        if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
            subset_lengths: List[int] = []
            for i, frac in enumerate(lengths):
                if frac < 0 or frac > 1:
                    raise ValueError(f"Fraction at index {i} is not between 0 and 1")
                n_items_in_split = int(math.floor(len(dataset) * frac))  # type: ignore[arg-type]
                subset_lengths.append(n_items_in_split)
            remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
            # add 1 to all the lengths in round-robin fashion until the remainder is 0
            for i in range(remainder):
                idx_to_add_at = i % len(subset_lengths)
                subset_lengths[idx_to_add_at] += 1
            lengths = subset_lengths
            for i, length in enumerate(lengths):
                if length == 0:
                    warnings.warn(f"Length of split at index {i} is 0. " f"This might result in an empty dataset.")

        # Cannot verify that dataset is Sized
        if sum(lengths) != len(dataset):  # type: ignore[arg-type]
            raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

        indices = generator.permutation(sum(lengths)).tolist()  # type: ignore[call-overload]
        return [
            TimeSeriesTrainDatasetSubset(dataset, indices[offset - length : offset])
            for offset, length in zip(_accumulate(lengths), lengths)
        ]
