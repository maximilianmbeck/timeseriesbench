# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torchtext
from datasets import DatasetDict, Value, load_dataset
from torch import nn
from torch.nn import functional as F

LOGGER = logging.getLogger(__name__)

from ....interfaces import Tokenizer
from ..lra_base import LraTextSequenceDatasetConfig, LraTextSequenceDatasetGenerator


@dataclass
class AanConfig(LraTextSequenceDatasetConfig):
    context_length: int = 4000  # set to l_max default from safari


class AanDatasetGenerator(LraTextSequenceDatasetGenerator):

    """
    # TODO - add description

    """

    shortname = "aan"
    config_class = AanConfig

    output_classes = 2

    def __init__(self, config: AanConfig):
        super().__init__(config)
        self.config = config
        self.generate_dataset()

    @property
    def output_dim(self) -> Sequence[int]:
        return (self.output_classes,)

    @property
    def context_length(self):
        return self.config.context_length

    @property
    def cache_name(self) -> str:
        return f"{self.shortname}/ctx_len-{self.config.context_length}-append_bos-{self.config.append_bos}-append_eos-{self.config.append_eos}"

    @property
    def prepared_data_dir(self) -> Path:
        return self.lra_prepared_data_dir / "aan"

    def get_input_target_key_mapping(self) -> tuple[Sequence[str], Sequence[str]]:
        return ["input_ids1", "input_ids2"], ["label"]

    def _preprocess_dataset(self) -> tuple[DatasetDict, Tokenizer]:
        """
        Preprocessing of this dataset consists of the following steps:
        1. Load the dataset from disk
        2. Remove unnecessary columns, cast to correct types
        3. convert the strings to list of characters and truncate to the desired length
        4. Generate vocabulary of all characters in the dataset
        5. Tokenize the dataset / characters to integers and add bos and eos tokens
        """
        # 1. Load the dataset from disk
        LOGGER.info("DATA PREP STEP (1/5): Loading raw data")
        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.prepared_data_dir / "new_aan_pairs.train.tsv"),
                "val": str(self.prepared_data_dir / "new_aan_pairs.eval.tsv"),
                "test": str(self.prepared_data_dir / "new_aan_pairs.test.tsv"),
            },
            delimiter="\t",
            column_names=["label", "input1_id", "input2_id", "text1", "text2"],
            keep_in_memory=True,
        )
        # 2. Remove unnecessary columns, cast to correct types
        LOGGER.info("DATA PREP STEP (2/5): Removing unnecessary columns and casting to correct types")
        dataset = dataset.remove_columns(["input1_id", "input2_id"])
        new_features = dataset["train"].features.copy()
        new_features["label"] = Value("int32")
        dataset = dataset.cast(new_features)

        # 3. convert the strings to list of characters and truncate to the desired length
        LOGGER.info("DATA PREP STEP (3/5): Converting strings to list of characters and truncating to desired length")
        tokenizer = list  # Just convert a string to a list of chars
        # Account for <bos> and <eos> tokens
        l_max = self.config.context_length - int(self.config.append_bos) - int(self.config.append_eos)
        tokenize = lambda example: {
            "tokens1": tokenizer(example["text1"])[:l_max],
            "tokens2": tokenizer(example["text2"])[:l_max],
        }
        dataset = dataset.map(
            tokenize,
            remove_columns=["text1", "text2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.config.num_workers, 1),
        )

        # 4. Generate vocabulary of all characters in the dataset
        LOGGER.info("DATA PREP STEP (4/5): Generating vocabulary of all characters in the dataset")
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens1"] + dataset["train"]["tokens2"],
            specials=(
                ["<pad>", "<unk>"]
                + (["<bos>"] if self.config.append_bos else [])
                + (["<eos>"] if self.config.append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        # 5. Tokenize the dataset / characters to integers and add bos and eos tokens
        LOGGER.info(
            "DATA PREP STEP (5/5): Tokenizing the dataset / characters to integers and adding bos and eos tokens"
        )
        encode = lambda text: vocab(
            (["<bos>"] if self.config.append_bos else []) + text + (["<eos>"] if self.config.append_eos else [])
        )
        numericalize = lambda example: {
            "input_ids1": encode(example["tokens1"]),
            "input_ids2": encode(example["tokens2"]),
        }
        tokenized_dataset = dataset.map(
            numericalize,
            remove_columns=["tokens1", "tokens2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.config.num_workers, 1),
        )

        return tokenized_dataset, vocab

    @property
    def collate_fn(self) -> Callable[[Any], Any]:
        def _collate_fn(batch):
            xs1, xs2, ys = zip(*[(data["input_ids1"], data["input_ids2"], data["label"]) for data in batch])
            lengths1 = torch.tensor([len(x) for x in xs1])
            lengths2 = torch.tensor([len(x) for x in xs2])
            xs1 = nn.utils.rnn.pad_sequence(xs1, padding_value=self.tokenizer["<pad>"], batch_first=True)
            xs2 = nn.utils.rnn.pad_sequence(xs2, padding_value=self.tokenizer["<pad>"], batch_first=True)
            # Pad both to same length
            # Shape (batch, length)
            L = max(xs1.size(1), xs2.size(1))
            xs1 = F.pad(xs1, (0, L - xs1.size(1)), value=self.tokenizer["<pad>"])
            xs2 = F.pad(xs2, (0, L - xs2.size(1)), value=self.tokenizer["<pad>"])
            ys = torch.tensor(ys)
            # return xs1, xs2, ys, lengths1, lengths2

            # Concatenate two batches
            xs = torch.cat([xs1, xs2], dim=0)
            lengths = torch.cat([lengths1, lengths2], dim=0)
            return xs, ys, {"lengths": lengths}

        return _collate_fn
