# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck


import logging
import pickle
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import torch
import torchmetrics
from datasets import DatasetDict
from torch import nn

from ...interfaces import SequenceConfigInterface, Tokenizer
from ...metrics import create_accuracy_metric
from ..basedataset import BaseSequenceDataset, BaseSequenceDatasetGenerator

LOGGER = logging.getLogger(__name__)

# Interface for tokenized dataset (resembles HuggingFace's Dataset)
InputtokensTargetDataset = Sequence[dict[str, Union[int, Sequence[int]]]]


# TODO take care of padding
# see https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
# Possible solution: Addd a get collate_fn method to the dataset generator
# returns the default, but is overridden in this dataset generator
# also make sure that the loss is applied to the correct sequence length


class TextSequenceDataset(BaseSequenceDataset):
    def __init__(
        self,
        dataset: InputtokensTargetDataset,
        vocab_tokenizer: Tokenizer,
        config: SequenceConfigInterface,
        input_keys: Sequence[str] = ["input_ids"],
        target_keys: Sequence[str] = ["Target"],
    ):
        super().__init__()
        self.config = config
        self.dataset: InputtokensTargetDataset = dataset
        self.vocab_tokenizer: Tokenizer = vocab_tokenizer  # vocabulary or tokenizer
        self.input_keys = input_keys
        self.target_keys = target_keys

    @property
    def vocab_size(self):
        return len(self.vocab_tokenizer)

    @property
    def context_length(self):
        return self.config.context_length

    @property
    def output_dim(self) -> Sequence[int]:
        return self.config.output_dim

    def __getitem__(self, index: int) -> tuple[Sequence[torch.Tensor], torch.Tensor]:
        # if only one input key or target key is given, return only the corresponding value
        # if multiple values are given, return a dict
        item = self.dataset[index]
        if len(self.input_keys) == 1 and len(self.target_keys) == 1:
            return item[self.input_keys[0]], item[self.target_keys[0]]
        return item

    def __len__(self) -> int:
        return len(self.dataset)


@dataclass
class TextSequenceDatasetConfig:
    data_dir: str = "./data"
    append_bos: bool = False
    append_eos: bool = True
    load_from_cache: bool = True
    save_to_cache: bool = True
    num_workers: int = 4  # only for tokenization


class TextSequenceDatasetGenerator(BaseSequenceDatasetGenerator):
    def __init__(self, config: TextSequenceDatasetConfig):
        super().__init__()
        self.config = config
        self._dataset_generated = False

        self._raw_datasets: DatasetDict = {}
        self._vocab_tokenizer: Tokenizer = None
        self._datasets: dict[str, TextSequenceDataset] = {}

    @abstractmethod
    def _generate_raw_datasets_and_vocab(self) -> tuple[DatasetDict, Tokenizer]:
        pass

    @property
    @abstractmethod
    def cache_name(self) -> str:
        pass

    @abstractmethod
    def get_input_target_key_mapping(self) -> tuple[Sequence[str], Sequence[str]]:
        pass

    @property
    def cache_dir(self) -> Path:
        return Path(self.config.data_dir) / self.cache_name

    def _load_raw_datasets_and_vocab(self, cache_dir: Path) -> tuple[DatasetDict, Tokenizer]:
        LOGGER.info(f"Loading raw datasets and vocab from {cache_dir}")
        with open(cache_dir / "vocab_tokenizer.pkl", "rb") as f:
            vocab_tokenizer = pickle.load(f)
        raw_datasets = DatasetDict.load_from_disk(cache_dir / "raw_datasets")
        return raw_datasets, vocab_tokenizer

    def _save_raw_datasets_and_vocab(
        self, cache_dir: Path, raw_datasets: DatasetDict, vocab_tokenizer: Tokenizer
    ) -> None:
        # TODO add file lock
        LOGGER.info(f"Saving raw datasets and vocab to {cache_dir}")
        with open(cache_dir / "vocab_tokenizer.pkl", "wb") as f:
            pickle.dump(vocab_tokenizer, f)
        raw_datasets.save_to_disk(cache_dir / "raw_datasets")

    def _get_raw_datasets_and_vocab(self) -> tuple[DatasetDict, Tokenizer]:
        if self.config.load_from_cache and self.cache_dir.exists():
            raw_datasets, vocab_tokenizer = self._load_raw_datasets_and_vocab(cache_dir=self.cache_dir)
        else:
            LOGGER.info("Generating raw datasets and vocab")
            raw_datasets, vocab_tokenizer = self._generate_raw_datasets_and_vocab()
            if self.config.save_to_cache:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._save_raw_datasets_and_vocab(
                    cache_dir=self.cache_dir, raw_datasets=raw_datasets, vocab_tokenizer=vocab_tokenizer
                )
        return raw_datasets, vocab_tokenizer

    def generate_dataset(self) -> None:
        self._raw_datasets, self._vocab_tokenizer = self._get_raw_datasets_and_vocab()
        input_key, target_key = self.get_input_target_key_mapping()
        self._raw_datasets.set_format("torch")  # use torch tensors
        self._datasets = {
            split: TextSequenceDataset(
                dataset=raw_dataset_split,
                vocab_tokenizer=self._vocab_tokenizer,
                config=self.config,
                input_keys=input_key,
                target_keys=target_key,
            )
            for split, raw_dataset_split in self._raw_datasets.items()
        }
        self._dataset_generated = True

    @property
    def tokenizer(self):
        return self._vocab_tokenizer

    @property
    def vocab_size(self):
        return len(self._vocab_tokenizer)

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

    @property
    def dataset_generated(self) -> bool:
        return self._dataset_generated

    @property
    def train_split(self) -> TextSequenceDataset:
        assert self.dataset_generated, "Dataset not generated yet."
        return self._datasets["train"]

    @property
    def validation_split(self) -> TextSequenceDataset:
        assert self.dataset_generated, "Dataset not generated yet."
        return self._datasets["val"]

    @property
    def test_split(self) -> Optional[TextSequenceDataset]:
        assert self.dataset_generated, "Dataset not generated yet."
        return self._datasets.get("test", None)

    @property
    def collate_fn(self) -> Callable[[Any], Any]:
        """Return a collate function that pads the input sequences to the same length."""

        # TODO maybe make sure that <pad> token exists in the vocab or add it if not or make configurable
        def _collate_fn(batch: list[tuple[Sequence[int], int]]) -> tuple[torch.Tensor, torch.Tensor]:
            xs, ys = zip(*batch)
            lengths = torch.tensor([len(x) for x in xs])
            xs = nn.utils.rnn.pad_sequence(xs, padding_value=self._vocab_tokenizer["<pad>"], batch_first=True)
            ys = torch.tensor(ys)
            return xs, ys, {"lengths": lengths}

        return _collate_fn
