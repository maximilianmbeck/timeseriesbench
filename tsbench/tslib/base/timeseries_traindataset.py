# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from abc import abstractmethod
from typing import Dict, Union

import numpy as np
import torch

from .base_dataset import BaseDataset

T_out = Union[torch.Tensor, np.ndarray]


class TimeSeriesTrainDataset(BaseDataset):
    """Base class for time series train datasets.
    This class defines the interface for training frameworks, i.e. provides
    the dimensions of the input and outputs."""

    @abstractmethod
    def __getitem__(self, index: int) -> Dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def target_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def context_length(self) -> int:
        """Returns the context length / time series length of the inputs."""
        pass

    @property
    @abstractmethod
    def target_length(self) -> int:
        """Returns time series length of the outputs."""
        pass
