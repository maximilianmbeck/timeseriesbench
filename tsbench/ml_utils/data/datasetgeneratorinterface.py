# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Optional, Union

from torch.utils import data
from torchmetrics import MetricCollection

from .split import subset
from .stateful_dataset import StatefulDataset


class DatasetGeneratorInterface(ABC):
    def generate_dataset(self) -> None:
        pass

    @property
    @abstractmethod
    def dataset_generated(self) -> bool:
        pass

    @property
    @abstractmethod
    def train_split(self) -> data.Dataset:
        pass

    @property
    @abstractmethod
    def validation_split(self) -> Optional[Union[data.Dataset, Mapping[str, data.Dataset]]]:
        pass

    @property
    def test_split(self) -> Optional[Union[data.Dataset, Mapping[str, data.Dataset]]]:
        return None

    @property
    def train_metrics(self) -> Optional[MetricCollection]:
        return None

    @property
    def validation_metrics(self) -> Optional[MetricCollection]:
        return None

    @property
    def test_metrics(self) -> Optional[MetricCollection]:
        return None

    @property
    def collate_fn(self) -> Optional[Callable[[Any], Any]]:
        """
        Returns:
            Optional[Callable[[Any], Any]]: Collate function for the dataset to be passed to dataloader.
                                            If None, default collate_fn is used.
        """
        return None


class DatasetGeneratorWrapper(DatasetGeneratorInterface):
    """
    Wrapper for a DatasetGenerator that implements the DatasetGeneratorInterface.
    This wrapper is used internally for creating the dataloders. To add new datasets users should implement
    the DatasetGeneratorInterface.
    It does the following:
    - It takes care that validation and testsplits always returns a dict of validation and test datasets.
    - It provides the option to wrap the training dataset with a stateful dataset.
    - It provides the option to subset the training, validation and test datasets.
    """

    def __init__(
        self,
        datasetgenerator: DatasetGeneratorInterface,
        stateful_train_dataset: bool = False,
        global_batch_size: int = -1,
        seed: int = -1,
        limit_n_train_samples: Optional[int] = None,
        limit_n_val_samples: Optional[int] = None,
        limit_n_test_samples: Optional[int] = None,
    ):
        assert isinstance(
            datasetgenerator, DatasetGeneratorInterface
        ), f"datasetgenerator must be of type DatasetGeneratorInterface, but got {type(datasetgenerator)}"
        assert not isinstance(datasetgenerator, DatasetGeneratorWrapper), "datasetgenerator must not be a wrapper"

        if stateful_train_dataset:
            assert global_batch_size > 0, f"global_batch_size must be > 0, but got {global_batch_size}"
            assert seed >= 0, f"seed must be >= 0, but got {seed}"

        self.datasetgenerator = datasetgenerator
        self.stateful_train_dataset = stateful_train_dataset
        self.global_batch_size = global_batch_size
        self.seed = seed
        self.limit_n_train_samples = limit_n_train_samples
        self.limit_n_val_samples = limit_n_val_samples
        self.limit_n_test_samples = limit_n_test_samples

    def generate_dataset(self) -> None:
        self.datasetgenerator.generate_dataset()

    @property
    def dataset_generated(self) -> bool:
        return self.datasetgenerator.dataset_generated

    @property
    def train_split(self) -> data.Dataset:
        train_ds = subset(dataset=self.datasetgenerator.train_split, n_samples=self.limit_n_train_samples)
        if self.stateful_train_dataset:
            train_ds = StatefulDataset(
                dataset=train_ds,
                global_batch_size=self.global_batch_size,
                batch_idx=0,
                seed=self.seed,
            )
        return train_ds

    @property
    def validation_split(self) -> Optional[Mapping[str, data.Dataset]]:
        val_ds = self.datasetgenerator.validation_split
        if isinstance(val_ds, data.Dataset):
            val_ds = {"0": subset(dataset=val_ds, n_samples=self.limit_n_val_samples)}
        else:
            val_ds = {k: subset(dataset=ds, n_samples=self.limit_n_val_samples) for k, ds in val_ds.items()}
        return val_ds

    @property
    def test_split(self) -> Optional[Mapping[str, data.Dataset]]:
        test_ds = self.datasetgenerator.test_split
        if isinstance(test_ds, data.Dataset):
            test_ds = {"0": subset(dataset=test_ds, n_samples=self.limit_n_test_samples)}
        else:
            test_ds = {k: subset(dataset=ds, n_samples=self.limit_n_test_samples) for k, ds in test_ds.items()}
        return test_ds

    @property
    def train_metrics(self) -> Optional[MetricCollection]:
        return self.datasetgenerator.train_metrics

    @property
    def validation_metrics(self) -> Optional[MetricCollection]:
        return self.datasetgenerator.validation_metrics

    @property
    def test_metrics(self) -> Optional[MetricCollection]:
        return self.datasetgenerator.test_metrics

    @property
    def collate_fn(self) -> Optional[Callable[[Any], Any]]:
        return self.datasetgenerator.collate_fn
