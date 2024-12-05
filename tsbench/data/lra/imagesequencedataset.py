# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck


from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
import torchmetrics
from torch.utils import data

from ...interfaces import SequenceConfigInterface
from ...metrics import create_accuracy_metric
from ...ml_utils.data.split import RandomSplitConfig, random_split_train_tasks
from ..basedataset import BaseSequenceDataset, BaseSequenceDatasetGenerator
from .datasettransformer import DatasetTransformer

# Support the sequential processing of image datasets
# use grayscale
# Options:
#   - tokenize or not (tokenization only possible for greyscale)
#   - use grayscale or rgb
#   - if not using grayscale: input channels sequentially or as 3-dimensional input sequence

Transform = Callable[[torch.Tensor], torch.Tensor]


class ImageSequenceDataset(BaseSequenceDataset):
    def __init__(self, dataset: DatasetTransformer, config: SequenceConfigInterface):
        self.config = config
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[index]

    @property
    def output_dim(self) -> Sequence[int]:
        return self.config.output_dim

    @property
    def input_dim(self) -> Sequence[int]:
        return self.config.input_dim

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def context_length(self) -> int:
        return self.config.context_length


@dataclass
class ImageSequenceDatasetConfig:
    data_dir: str = "./data"
    split: RandomSplitConfig = field(default_factory=RandomSplitConfig)


class ImageSequenceDatasetGenerator(BaseSequenceDatasetGenerator):
    def __init__(self, config: ImageSequenceDatasetConfig):
        self.config = config

        self._raw_datasets: dict[str, data.Dataset] = {}
        self._datasets: dict[str, ImageSequenceDataset] = {}
        self._dataset_generated = False

    @abstractmethod
    def _load_raw_datasets(self) -> dict[str, data.Dataset]:
        pass

    @abstractmethod
    def _create_image_transforms(self) -> tuple[list[Transform], list[Transform]]:
        pass

    def generate_dataset(self) -> None:
        self._raw_datasets = self._load_raw_datasets()
        if not "val" in self._raw_datasets:
            self._raw_datasets["train"], self._raw_datasets["val"] = random_split_train_tasks(
                self._raw_datasets["train"], self.config.split
            )

        assert all(x in self._raw_datasets for x in ["train", "val"]), "Missing train or val split."

        train_transforms, val_transforms = self._create_image_transforms()

        dataset_config = SequenceConfigInterface(
            context_length=self.context_length,
            output_dim=self.output_dim,
            input_dim=self.input_dim,
            vocab_size=self.vocab_size,
        )

        self._datasets["train"] = ImageSequenceDataset(
            DatasetTransformer(self._raw_datasets["train"], image_transforms=train_transforms), config=dataset_config
        )
        self._datasets["val"] = ImageSequenceDataset(
            DatasetTransformer(self._raw_datasets["val"], image_transforms=val_transforms), config=dataset_config
        )
        if "test" in self._raw_datasets:
            self._datasets["test"] = ImageSequenceDataset(
                DatasetTransformer(self._raw_datasets["test"], image_transforms=val_transforms), config=dataset_config
            )

        self._dataset_generated = True

    @property
    def dataset_generated(self) -> bool:
        return self._dataset_generated

    @property
    def train_split(self) -> ImageSequenceDataset:
        assert self.dataset_generated, "Dataset not generated yet."
        return self._datasets["train"]

    @property
    def validation_split(self) -> ImageSequenceDataset:
        assert self.dataset_generated, "Dataset not generated yet."
        return self._datasets["val"]

    @property
    def test_split(self) -> Optional[ImageSequenceDataset]:
        assert self.dataset_generated, "Dataset not generated yet."
        return self._datasets.get("test", None)

    def _create_accuracy_metric(self) -> torchmetrics.Metric:
        output_dim = self.output_dim
        assert len(output_dim) == 1, "Only support single output dimension."
        return create_accuracy_metric(num_classes=output_dim[0])

    @property
    def train_metrics(self) -> torchmetrics.MetricCollection:
        return torchmetrics.MetricCollection([self._create_accuracy_metric()])

    @property
    def validation_metrics(self) -> torchmetrics.MetricCollection:
        return torchmetrics.MetricCollection([self._create_accuracy_metric()])

    @property
    def test_metrics(self) -> torchmetrics.MetricCollection:
        return torchmetrics.MetricCollection([self._create_accuracy_metric()])
