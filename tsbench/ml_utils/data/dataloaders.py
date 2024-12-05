# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck


import copy
import logging
from dataclasses import asdict
from typing import Callable, Mapping, Optional, Tuple

from torch.utils import data

from ...config import Config, DataConfig
from ..ddp_utils.setup import is_distributed
from .datasetgeneratorinterface import (
    DatasetGeneratorInterface,
    DatasetGeneratorWrapper,
)

LOGGER = logging.getLogger(__name__)


def get_dataloader_creator(
    datasetgenerator: DatasetGeneratorInterface,
    config: Optional[Config] = None,
    data_cfg: Optional[DataConfig] = None,
    distributed: Optional[bool] = None,
    distributed_validation: Optional[bool] = None,
    seed: Optional[int] = None,
) -> Callable[[], Tuple[data.DataLoader, Mapping[str, data.DataLoader]]]:
    assert isinstance(
        datasetgenerator, DatasetGeneratorInterface
    ), f"datasetgenerator must be of type DatasetGeneratorInterface, but got {type(datasetgenerator)}"

    # * determine setup variables
    # determine if distributed training is enabled
    if distributed is None:
        distributed = config.ddp is not None and config.ddp.enable_ddp and is_distributed()
    if distributed_validation is None:
        distributed_validation = config.ddp is not None and config.ddp.distributed_validation

    if seed is None:
        seed = config.experiment_data.seed
    if data_cfg is None:
        data_cfg = config.data

    if config is not None:
        global_batch_size = config.global_batch_size
    else:
        global_batch_size = data_cfg.dl_kwargs.batch_size

    # * create dataloaders
    # wrap dataset generator
    datasetgenerator = DatasetGeneratorWrapper(
        datasetgenerator=datasetgenerator,
        stateful_train_dataset=data_cfg.stateful_train_dataset,
        global_batch_size=global_batch_size,
        seed=seed,
        limit_n_train_samples=data_cfg.limit_n_train_samples,
        limit_n_val_samples=data_cfg.limit_n_val_samples,
    )
    train_split = datasetgenerator.train_split
    val_splits = datasetgenerator.validation_split

    # setup train dataloader kwargs
    train_dl_kwargs = copy.deepcopy(data_cfg.dl_kwargs)
    if data_cfg.stateful_train_dataset:
        LOGGER.info(
            "Using a stateful train dataset, so setting shuffle=False and persistent_workers=False in train dataloader kwargs."
        )
        train_dl_kwargs.shuffle = False
        train_dl_kwargs.persistent_workers = False

    # setup val dataloader kwargs
    val_dl_kwargs = copy.deepcopy(data_cfg.dl_kwargs)
    val_dl_kwargs.shuffle = False
    val_dl_kwargs.drop_last = False

    if distributed:
        # train dataloader
        train_sampler = data.DistributedSampler(
            train_split, shuffle=train_dl_kwargs.shuffle, drop_last=train_dl_kwargs.drop_last, seed=seed
        )
        train_dl = data.DataLoader(
            train_split,
            batch_size=train_dl_kwargs.batch_size,
            sampler=train_sampler,
            num_workers=train_dl_kwargs.num_workers,
            pin_memory=train_dl_kwargs.pin_memory,
            drop_last=train_dl_kwargs.drop_last,
            persistent_workers=train_dl_kwargs.persistent_workers,
            collate_fn=datasetgenerator.collate_fn,
        )
        # val dataloader
        if distributed_validation:
            val_dl = {}
            for split_name, split in val_splits.items():
                val_sampler = data.DistributedSampler(split, shuffle=False, drop_last=False, seed=seed)
                val_dl[split_name] = data.DataLoader(
                    split,
                    batch_size=val_dl_kwargs.batch_size,
                    sampler=val_sampler,
                    num_workers=val_dl_kwargs.num_workers,
                    pin_memory=val_dl_kwargs.pin_memory,
                    drop_last=False,
                    persistent_workers=val_dl_kwargs.persistent_workers,
                    collate_fn=datasetgenerator.collate_fn,
                )
        else:
            val_dl = {
                split_name: data.DataLoader(split, **asdict(val_dl_kwargs), collate_fn=datasetgenerator.collate_fn)
                for split_name, split in val_splits.items()
            }
    else:
        train_dl = data.DataLoader(train_split, **asdict(train_dl_kwargs), collate_fn=datasetgenerator.collate_fn)
        val_dl = {
            split_name: data.DataLoader(split, **asdict(val_dl_kwargs), collate_fn=datasetgenerator.collate_fn)
            for split_name, split in val_splits.items()
        }
    return lambda: (train_dl, val_dl)


def create_val_loaders(
    val_splits: Mapping[str, data.Dataset],
    collate_fn: Optional[Callable] = None,
    config: Optional[Config] = None,
    data_cfg: Optional[DataConfig] = None,
    distributed: Optional[bool] = None,
    seed: Optional[int] = None,
) -> Mapping[str, data.DataLoader]:
    # * determine setup variables
    # determine if distributed training is enabled
    if distributed is None:
        distributed = (
            config.ddp is not None and config.ddp.enable_ddp and is_distributed() and config.ddp.distributed_validation
        )

    if seed is None:
        seed = config.experiment_data.seed
    if data_cfg is None:
        data_cfg = config.data

    # setup val dataloader kwargs
    val_dl_kwargs = copy.deepcopy(data_cfg.dl_kwargs)
    val_dl_kwargs.shuffle = False
    val_dl_kwargs.drop_last = False

    if distributed:
        val_dl = {}
        for split_name, split in val_splits.items():
            val_sampler = data.DistributedSampler(split, shuffle=False, drop_last=False, seed=seed)
            val_dl[split_name] = data.DataLoader(
                split,
                batch_size=val_dl_kwargs.batch_size,
                sampler=val_sampler,
                num_workers=val_dl_kwargs.num_workers,
                pin_memory=val_dl_kwargs.pin_memory,
                drop_last=False,
                persistent_workers=val_dl_kwargs.persistent_workers,
                collate_fn=collate_fn,
            )
    else:
        val_dl = {
            split_name: data.DataLoader(split, **asdict(val_dl_kwargs), collate_fn=collate_fn)
            for split_name, split in val_splits.items()
        }

    return val_dl


def create_train_loader(
    train_split: data.Dataset,
    collate_fn: Optional[Callable] = None,
    config: Optional[Config] = None,
    data_cfg: Optional[DataConfig] = None,
    distributed: Optional[bool] = None,
    seed: Optional[int] = None,
) -> data.DataLoader:
    # * determine setup variables
    # determine if distributed training is enabled
    if distributed is None:
        distributed = config.ddp is not None and config.ddp.enable_ddp and is_distributed()

    if seed is None:
        seed = config.experiment_data.seed
    if data_cfg is None:
        data_cfg = config.data

    # setup train dataloader kwargs
    train_dl_kwargs = copy.deepcopy(data_cfg.dl_kwargs)
    if data_cfg.stateful_train_dataset:
        LOGGER.debug(
            "Using a stateful train dataset, so setting shuffle=False and persistent_workers=False in train dataloader kwargs."
        )
        train_dl_kwargs.shuffle = False
        train_dl_kwargs.persistent_workers = False

    if distributed:
        # train dataloader
        train_sampler = data.DistributedSampler(
            train_split, shuffle=train_dl_kwargs.shuffle, drop_last=train_dl_kwargs.drop_last, seed=seed
        )
        train_dl = data.DataLoader(
            train_split,
            batch_size=train_dl_kwargs.batch_size,
            sampler=train_sampler,
            num_workers=train_dl_kwargs.num_workers,
            pin_memory=train_dl_kwargs.pin_memory,
            drop_last=train_dl_kwargs.drop_last,
            persistent_workers=train_dl_kwargs.persistent_workers,
            collate_fn=collate_fn,
        )
    else:
        train_dl = data.DataLoader(train_split, **asdict(train_dl_kwargs), collate_fn=collate_fn)

    return train_dl
