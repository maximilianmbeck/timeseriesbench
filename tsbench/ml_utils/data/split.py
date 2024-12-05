# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils import data


@dataclass
class RandomSplitConfig:
    num_train_tasks: int = 1
    train_task_idx: int = 0
    train_val_split: float = 0.9
    seed: int = 0
    num_subsplit_tasks: int = 0
    subsplit_first_n_train_tasks: int = 0
    restrict_n_samples_train_task: int = -1
    restrict_n_samples_val_task: int = -1


def random_split_train_tasks(
    dataset: data.Dataset,
    config: RandomSplitConfig,
) -> Tuple[data.Dataset, data.Dataset]:
    """Splits a dataset into different (sample-wise) training tasks.
    Each training task has different set of data samples. Validation set is same for every task.
    It further allows to subsplit the first `subsplit_n_train_tasks` further into `num_subsplit_tasks`.

    Args:
        dataset (data.Dataset): The dataset to split.
        num_train_tasks (int, optional): Number of training tasks to split. Defaults to 1.
        train_task_idx (int, optional): The current training task. Defaults to 0.
        train_val_split (float, optional): Fraction of train/val samples. Defaults to 0.9.
        seed (int, optional): The seed. Defaults to 0.
        num_subsplit_tasks (int, optional): Number of subsplit tasks. Defaults to 0.
        subsplit_first_n_train_tasks (int, optional): The number of first training tasks to further subsplit. Defaults to 0.
        restrict_n_samples_train_task (int, optional): Restrict the number of samples in the train split. If -1, use the whole train split. Defaults to -1.
        restrict_n_samples_val_task (int, optional): Restrict the number of samples in the val split. If -1, use the whole val split. Defaults to -1.

    Returns:
        Tuple[data.Dataset, data.Dataset]: train dataset, val dataset
    """
    cfg = config
    assert (
        cfg.train_task_idx >= 0
        and cfg.train_task_idx < (cfg.num_train_tasks - cfg.subsplit_first_n_train_tasks) + cfg.num_subsplit_tasks
    ), "Invalid train_task_idx given."

    n_train_samples = int(cfg.train_val_split * len(dataset))

    n_samples_per_task = int(n_train_samples / cfg.num_train_tasks)

    train_split_lengths = cfg.num_train_tasks * [n_samples_per_task]

    # make sure that sum of all splits equal total number of samples in dataset
    # n_val_samples can be greater than specified by train_val_split
    n_val_samples = len(dataset) - torch.tensor(train_split_lengths).sum().item()

    split_lengths = cfg.num_train_tasks * [n_samples_per_task] + [n_val_samples]
    data_splits = data.random_split(dataset, split_lengths, generator=torch.Generator().manual_seed(cfg.seed))

    if cfg.num_subsplit_tasks > 0:
        # further split first ´subsplit_first_n_train_tasks´ into `num_subsplit_tasks`
        subsplit_dataset = data.ConcatDataset(data_splits[: cfg.subsplit_first_n_train_tasks])
        # remove first n train tasks idxs from data split list
        data_splits = data_splits[cfg.subsplit_first_n_train_tasks :]
        n_samples_per_subsplit = int(len(subsplit_dataset) / cfg.num_subsplit_tasks)

        subsplit_lengths = cfg.num_subsplit_tasks * [n_samples_per_subsplit]
        # distribute remaining samples (due to rounding) from beginning
        samples_remaining = len(subsplit_dataset) - sum(subsplit_lengths)
        for i in range(len(subsplit_lengths)):
            if samples_remaining <= 0:
                break
            subsplit_lengths[i] += 1
            samples_remaining -= 1

        assert sum(subsplit_lengths) == len(subsplit_dataset)

        data_subsplits = data.random_split(
            subsplit_dataset, subsplit_lengths, generator=torch.Generator().manual_seed(cfg.seed + 1)
        )

        # concat data_splits: [subsplit sets] + train sets + val set
        data_splits = data_subsplits + data_splits

    # select train task split + val split
    train_set, val_set = data_splits[cfg.train_task_idx], data_splits[-1]

    # restrict number of training samples
    if cfg.restrict_n_samples_train_task > 0:
        assert (
            len(train_set) >= cfg.restrict_n_samples_train_task
        ), f"Not enough number of samples in the training set! Trying to restrict to {cfg.restrict_n_samples_train_task} samples, but training set has only {len(train_set)} samples."

        train_set = data.Subset(train_set, range(cfg.restrict_n_samples_train_task))

    # restrict number of validation samples
    if cfg.restrict_n_samples_val_task > 0:
        assert (
            len(val_set) >= cfg.restrict_n_samples_val_task
        ), f"Not enough number of samples in the training set! Trying to restrict to {cfg.restrict_n_samples_val_task} samples, but training set has only {len(val_set)} samples."

        val_set = data.Subset(val_set, range(cfg.restrict_n_samples_val_task))

    return train_set, val_set


def subset(dataset: data.Dataset, n_samples: Optional[int] = None, mode: str = "first") -> data.Dataset:
    if n_samples is None:
        return dataset

    if mode == "first":
        indices = torch.arange(n_samples)
    elif mode == "random":
        indices = torch.randperm(len(dataset))[:n_samples]
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return data.Subset(dataset, indices)
