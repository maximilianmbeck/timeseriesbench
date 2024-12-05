# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from torch.utils.data import Dataset

from ..interfaces import SequenceInterface, TokenizerInterface
from ..ml_utils.data.datasetgeneratorinterface import DatasetGeneratorInterface


class BaseSequenceDatasetGenerator(SequenceInterface, TokenizerInterface, DatasetGeneratorInterface):
    """Interface for sequence dataset generators.
    Note: We add the TokenizerInterface because a general Sequence dataset might be
    a language dataset with a tokenizer (e.g. some tasks in LRA)."""

    pass


class BaseSequenceDataset(SequenceInterface, TokenizerInterface, Dataset):
    """Interface for sequence datasets."""

    pass
