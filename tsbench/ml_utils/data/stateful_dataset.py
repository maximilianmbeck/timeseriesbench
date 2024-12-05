# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck, Markus Spanring

import logging

import numpy as np
from torch.utils import data

from ..utils import call_on_wrapped_object

LOGGER = logging.getLogger(__name__)


class StatefulDataset(data.Dataset):
    """Wrapper around a generic pytorch dataset to resume from an intra-epoch step.
    This wrapper takes care of shuffling the dataset and setting the correct seed for the random number generator.

    NOTES:
    - If you are using this wrapper in distributed mode, make sure that you provide the correct global_batch_size depending
      on the number of GPUs you are using.
    - Also make sure that you disable all other shuffling mechanisms (e.g. in the dataloader), otherwise the data order will
      not be the same.

    train progress:
    - checkpoint is saved before progress idx is updated
    idx=1 batch update [save_checkpoint=1] idx=2 batch update [save_checkpoint=2] [load_checkpoint=2] idx=3...
                                                                                  ^
                                                                                  At this point we have seen 2 batches.
                                                                                  These are excluded with this wrapper.
    Args:
        dataset (data.Dataset): The dataset to wrap
        global_batch_size (int): The global batch size, i.e. the batch size multiplied by the number of gpus and number of gradient accumulation steps.
        batch_idx (int, optional): The batch index to start from (inclusive, i.e. we will see this batch_idx during training). Defaults to 0.
        seed (int, optional): The seed for the random number generator. Defaults to 0.
    """

    def __init__(self, dataset: data.Dataset, global_batch_size: int, batch_idx: int = 0, seed: int = 0):
        self.dataset = dataset
        self._global_batch_size = global_batch_size
        self.seed = seed
        self._epoch = 0
        self.set_epoch(0, batch_idx)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def epoch_seed(self) -> int:
        return self.epoch + self.seed

    def set_epoch(self, epoch: int, batch_idx: int = 0) -> None:
        assert epoch >= 0, f"epoch must be >= 0, but got {epoch}"
        assert batch_idx >= 0, f"batch_idx must be >= 0, but got {batch_idx}"
        if batch_idx > 0:
            LOGGER.warning(
                f"You are starting from a batch_idx = {batch_idx} > 0. This means that you are not starting from the beginning of the epoch."
            )

        self._epoch = epoch
        rng = np.random.default_rng(self.epoch_seed)

        # self.dataset could be a wrapper itself, e.g. a Subset. In this case we need to call set_epoch on the wrapped object.
        call_on_wrapped_object(
            wrapper=self.dataset,
            wrapped_obj_name="dataset",
            method="set_epoch",
            kwargs=dict(epoch=epoch),
            error_when_not_called=False,
        )

        self.index = list(rng.permutation(len(self.dataset)))

        assert batch_idx * self._global_batch_size < len(
            self.index
        ), f"There are only {len(self.index) // self._global_batch_size} train steps. You wanted to start at {batch_idx}"
        self.index = self.index[batch_idx * self._global_batch_size :]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        return self.dataset[self.index[idx]]
