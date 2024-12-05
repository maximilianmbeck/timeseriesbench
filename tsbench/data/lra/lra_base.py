# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck


from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

from datasets import DatasetDict

from ...interfaces import Tokenizer
from .download_prepare import (
    LRA_DOWNLOAD_FOLDER,
    LRA_PREPARED_FOLDER,
    download_and_prepare_lra_data,
)
from .textsequencedataset import TextSequenceDatasetConfig, TextSequenceDatasetGenerator

# This file contains the base class for all LRA text datasets, where the data is from official LRA sources.
# this class should take care of the following:
# - loading or downloading the lra data

# the dataset preprocessing is delegated to respective child classes


@dataclass
class LraTextSequenceDatasetConfig(TextSequenceDatasetConfig):
    redownload: bool = False
    delete_downloaded: bool = True


class LraTextSequenceDatasetGenerator(TextSequenceDatasetGenerator):
    def __init__(self, config: LraTextSequenceDatasetConfig):
        super().__init__(config=config)
        self.config = config
        self._data_dir = Path(self.config.data_dir)

    @abstractmethod
    def _preprocess_dataset(self) -> tuple[DatasetDict, Tokenizer]:
        pass

    @property
    @abstractmethod
    def prepared_data_dir(self) -> Path:
        pass

    @property
    def lra_prepared_data_dir(self) -> Path:
        return self._data_dir / LRA_PREPARED_FOLDER

    @property
    def lra_downloaded_data_dir(self) -> Path:
        return self._data_dir / LRA_DOWNLOAD_FOLDER

    @property
    def is_data_prepared(self) -> bool:
        return self.prepared_data_dir.exists()

    def _generate_raw_datasets_and_vocab(self) -> tuple[DatasetDict, Tokenizer]:
        if not self.is_data_prepared or self.config.redownload:
            download_and_prepare_lra_data(
                data_dir=self._data_dir,
                delete_downloaded=self.config.delete_downloaded,
                overwrite_existing=self.config.redownload,
            )
        processed_dataset, vocab_tokenizer = self._preprocess_dataset()
        return processed_dataset, vocab_tokenizer
