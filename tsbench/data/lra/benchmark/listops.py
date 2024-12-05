# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torchtext
from datasets import DatasetDict, load_dataset
from tqdm import tqdm

from ....interfaces import Tokenizer
from ..lra_base import LraTextSequenceDatasetConfig, LraTextSequenceDatasetGenerator

LOGGER = logging.getLogger(__name__)


# LRA tokenizer renames ']' to 'X' and delete parentheses as their tokenizer removes
# non-alphanumeric characters.
# https://github.com/google-research/long-range-arena/blob/264227cbf9591e39dd596d2dc935297a2070bdfe/lra_benchmarks/listops/input_pipeline.py#L46
def listops_processor(s):
    return s.translate({ord("]"): ord("X"), ord("("): None, ord(")"): None}).split()


@dataclass
class ListOpsConfig(LraTextSequenceDatasetConfig):
    context_length: int = 2048


class ListOpsDatasetGenerator(LraTextSequenceDatasetGenerator):
    """The ListOps dataset aims to test models on hierarchically structured data in a long-context scenario.
    It has a sequence length of up to 2K (2048) and is a ten-way classification task. The raw data is stored in .tsv format.
    The sequences have variable length.

    The raw data looks as follows:
    ```
    ( ( ( ( ( ( ( ( [MAX 1 ) ( ( ( ( ( [MIN 3 ) ( ( ( [MED 6 ) 1 ) ] ) ) 4 ) 5 ) ] ) ) 0 ) ( ( ( ( ( ( [SM ( ( ( ( ( ( ( ( ( [MED 1 ) 6 ) 2 ) 1 ) 0 ) 8 ) 2 ) 3 ) ] ...
    ```

    During preprocessing the parentheses (`(` and `)`) are removed and the `]` is replaced by `X`.

    The processed data looks as follows:
    ```
    ['[MAX', '1', '[MIN', '3', '[MED', '6', '1', 'X', '4', '5', 'X', '0', '[SM', '[MED', '1', '6', '2', '1', '0', '8', '2', '3', 'X' ...
    ```

    It will be tokenized into integers. The vocabulary is created from the training data. The vocabulary size is 18.

    """

    shortname = "listops"
    config_class = ListOpsConfig

    output_classes = 10

    def __init__(self, config: ListOpsConfig):
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
        return self.lra_prepared_data_dir / "listops"

    def get_input_target_key_mapping(self) -> tuple[Sequence[str], Sequence[str]]:
        return ["input_ids"], ["Target"]

    def _preprocess_dataset(self) -> tuple[DatasetDict, Tokenizer]:
        # * load raw data
        LOGGER.info("DATA PREP STEP (1/3): Loading raw data")
        raw_dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.prepared_data_dir / "basic_train.tsv"),
                "val": str(self.prepared_data_dir / "basic_val.tsv"),
                "test": str(self.prepared_data_dir / "basic_test.tsv"),
            },
            delimiter="\t",
            keep_in_memory=True,
            cache_dir=self.config.data_dir,
        )

        # * preprocess samples: Remove parentheses and replace ']' with 'X' and restrict length
        LOGGER.info("DATA PREP STEP (2/3): Preprocessing raw data")
        max_len = self.config.context_length - int(self.config.append_bos) - int(self.config.append_eos)

        def _preprocessor(example: dict[str, Any]) -> dict[str, Any]:
            return {"proc_source": listops_processor(example["Source"])[:max_len]}

        processed_dataset = raw_dataset.map(
            _preprocessor,
            remove_columns=["Source"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(1, self.config.num_workers),
        )

        # * tokenize samples: Add bos and eos token and tokenize
        LOGGER.info("DATA PREP STEP (3/3): Tokenizing preprocessed data")
        # build tokenizer / vocab from training data
        vocab = torchtext.vocab.build_vocab_from_iterator(
            tqdm(processed_dataset["train"]["proc_source"], desc="Building vocab", file=sys.stdout),
            min_freq=1,
            specials=["<pad>", "<unk>"]
            + (["<bos>"] if self.config.append_bos else [])
            + (["<eos>"] if self.config.append_eos else []),
        )
        vocab.set_default_index(vocab["<unk>"])

        def _tokenizer(example: dict[str, Any]) -> dict[str, Any]:
            return {
                "input_ids": vocab(
                    (["<bos>"] if self.config.append_bos else [])
                    + example["proc_source"]
                    + (["<eos>"] if self.config.append_eos else [])
                )
            }

        tokenized_dataset = processed_dataset.map(
            _tokenizer,
            remove_columns=["proc_source"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(1, self.config.num_workers),
        )
        return tokenized_dataset, vocab
