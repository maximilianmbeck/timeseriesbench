# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Type

import pandas as pd


class BaseDataset(ABC):
    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_meta_data_summary(
        self, meta_data_types: Tuple[Type] = (int, str, float), indices: List[int] = [], **kwargs
    ) -> pd.DataFrame:
        pass

    @property
    def keys(self) -> List[str]:
        """Return a list of all keys in the dataset. Indices correspond to the indices of the dataset."""
        return self.get_meta_data_summary(meta_data_types=(str))["key"].values.tolist()

    @property
    def root(self) -> Any:
        """Root directory of the dataset."""
        return None
