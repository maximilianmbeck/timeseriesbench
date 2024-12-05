# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from dataclasses import dataclass

import torch
import torchmetrics
from torch.utils import data
from torchmetrics import MetricCollection

from ..ml_utils.data.datasetgeneratorinterface import DatasetGeneratorInterface


@dataclass
class DummyTsClassificationDsGenConfig:
    num_classes: int = 10
    num_features: int = 15  # channels
    num_timesteps: int = 200
    num_samples_train: int = 1000
    num_samples_val: int = 100


class DummyTsClassificationDsGen(DatasetGeneratorInterface):
    config_class = DummyTsClassificationDsGenConfig

    def __init__(self, config: DummyTsClassificationDsGenConfig = DummyTsClassificationDsGenConfig()):
        self.config = config
        self._dataset_generated = False

    def _generate_tensordataset(self, num_samples: int):
        x = torch.randn(num_samples, self.config.num_features, self.config.num_timesteps)
        y = torch.randint(0, self.config.num_classes, (num_samples,))
        return data.TensorDataset(x, y)

    def generate_dataset(self) -> None:
        self._train_split = self._generate_tensordataset(self.config.num_samples_train)
        self._val_split = self._generate_tensordataset(self.config.num_samples_val)
        self._dataset_generated = True

    @property
    def dataset_generated(self) -> bool:
        return self._dataset_generated

    @property
    def train_split(self) -> data.Dataset:
        return self._train_split

    @property
    def val_split(self) -> data.Dataset:
        return self._val_split

    @property
    def train_metrics(self) -> MetricCollection:
        return torchmetrics.MetricCollection(
            [torchmetrics.classification.MulticlassAccuracy(num_classes=self.config.num_classes)]
        )

    @property
    def val_metrics(self) -> MetricCollection:
        return torchmetrics.MetricCollection(
            [torchmetrics.classification.MulticlassAccuracy(num_classes=self.config.num_classes)]
        )
