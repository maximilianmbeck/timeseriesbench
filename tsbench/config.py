# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck


from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from .ml_utils.config import DDPConfig, ExperimentConfig, NameAndKwargs, TrainerConfig


@dataclass
class DataloaderConfig:
    batch_size: int = 1
    num_workers: int = 2
    shuffle: bool = False
    pin_memory: bool = True
    drop_last: bool = True
    persistent_workers: bool = False


@dataclass
class DataConfig:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    dl_kwargs: DataloaderConfig = field(default_factory=DataloaderConfig)  # dataloader kwargs
    stateful_train_dataset: bool = True  # use stateful train dataset, allows resuming to train steps
    limit_n_train_samples: Optional[int] = None
    limit_n_val_samples: Optional[int] = None


@dataclass
class MetricConfig(NameAndKwargs):
    stage: Sequence[str] = field(default_factory=lambda: ["train", "validation"])


@dataclass
class Config:
    experiment_data: ExperimentConfig
    trainer: TrainerConfig
    model: NameAndKwargs
    data: DataConfig
    loss: NameAndKwargs
    wandb: bool = True
    metrics: list[MetricConfig] = field(default_factory=list)
    ddp: Optional[DDPConfig] = None

    @property
    def global_batch_size(self) -> int:
        """Returns the global batch size, i.e. the batch size multiplied by the number of gpus
        and number of gradient accumulation steps."""
        return self.data.dl_kwargs.batch_size * self.trainer.gradient_accumulation_steps * self.world_size

    @property
    def mini_batch_size(self) -> int:
        """(Mini) batch size is the batch passed through the model at once."""
        return self.data.dl_kwargs.batch_size

    @property
    def gradient_accumulation_steps(self) -> int:
        """Number of gradient accumulation steps."""
        return self.trainer.gradient_accumulation_steps

    @property
    def world_size(self) -> int:
        """Number of ranks (number of processes / GPUs used for training)."""
        if self.ddp is None or not self.ddp.enable_ddp:
            num_gpus = 1
        else:
            num_gpus = self.ddp.world_size
        return num_gpus

    @property
    def n_nodes(self) -> int:
        """Number of nodes involved in training."""
        if self.ddp is None or not self.ddp.enable_ddp:
            n_nodes = 1
        else:
            n_nodes = self.ddp.n_nodes
        return n_nodes
