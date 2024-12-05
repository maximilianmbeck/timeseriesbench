# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class NameAndKwargs:
    name: str
    kwargs: Optional[dict[str, Any]] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    entity: Optional[str]
    project_name: str
    experiment_tag: Optional[str]
    experiment_type: Optional[str]
    experiment_name: str
    experiment_dir: Optional[str]  # do not set, will be set during run initialization
    experiment_notes: Optional[str]
    seed: int
    gpu_id: Optional[int]
    job_name: Optional[str]
    hostname: Optional[str]
    # output_dir: is the directory where wandb outputs and run outputs are stored
    output_dir: Optional[str]  # if None, then use cwd() [current working directory]


@dataclass
class ResumeTrainingConfig:
    job_dir: str
    checkpoint_idx: int


@dataclass
class TrainingStrategyConfig:
    """Configuration for training precision.
    Default configuration is for single GPU training with float32 precision."""

    enable_mixed_precision: bool = False
    precision: str = "float32"  # 'float32', 'float16' or 'bfloat16'
    enable_autocast_gradscaler: bool = True  # passed to constructor of autocast and gradscaler
    precision_dtype: Optional[str] = None  # passed to constructor of autocast and gradscaler
    use_torch_compile: bool = False
    torch_compile_kwargs: Optional[dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        import torch

        dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        assert (
            self.precision in dtypes
        ), f"precision must be one of 'float32', 'float16' or 'bfloat16' but got {self.precision}"
        self.precision_dtype = dtypes[self.precision]


@dataclass
class DDPConfig:
    """
    Config for DistributedDataParallel (DDP) training.

    There are two options for DDP: Single-Node and Multi-Node training.

    Single-Node training [n_nodes=1]:
    The devices per node are specified in `devices` or the number of processes per node in `n_procs_per_node`.
    If `devices` is specified, then `n_procs_per_node` is ignored.
    If `devices` is not specified, then `n_procs_per_node` is used to determine the number of processes per node.
    It will use the first `n_procs_per_node` GPUs on the node.

    Multi-Node training [n_nodes>1]:
    Note: It is currently not possible to use different devices per node (e.g. 2 GPUs on node 1 and 4 GPUs on node 2).
    It is assumed that every node has the same number of GPUs.
    We do not use the "elastic" capabilities of torchrun (we do not set a redezvous backend).
    """

    # NOTE: in Multi-Node setup it is currently not possible to use different devices per node
    # (e.g. 2 GPUs on node 1 and 4 GPUs on node 2)
    # it is assumed that every node has the same number of GPUs

    n_nodes: int = 1
    n_procs_per_node: int = 1

    devices: str = ""
    backend: str = "nccl"
    init_method: str = "env://"
    enable_ddp: bool = True

    master_addr: str = "localhost"
    master_port: str = "12355"

    distributed_validation: bool = True

    static_graph: bool = False

    # filled in later
    # NOTE: all start with 0
    local_rank: Optional[int] = None
    global_rank: Optional[int] = None
    node_rank: Optional[int] = None

    @property
    def is_multi_node(self) -> bool:
        return self.n_nodes > 1

    @property
    def world_size(self) -> int:
        if self.is_multi_node:
            return self.n_nodes * self.n_procs_per_node
        else:
            if self.devices == "":
                return self.n_procs_per_node
            else:
                return len(self.devices.split(","))

    @property
    def ranks_per_node(self) -> int:
        if self.is_multi_node:
            return self.n_procs_per_node
        else:
            if self.devices == "":
                return self.n_procs_per_node
            else:
                return len(self.devices.split(","))

    def gpu_id_for_global_rank(self, global_rank: int) -> int:
        if self.is_multi_node:
            return global_rank % self.ranks_per_node
        else:
            if self.devices == "":
                return global_rank
            else:
                return int(self.devices.split(",")[global_rank])

    def get_ddp_summary(self) -> str:
        """Returns a string with a summary of the DDP configuration."""
        if self.is_multi_node:
            return f"[Multi-Node DDP] n_nodes={self.n_nodes}, n_procs_per_node={self.n_procs_per_node}, world_size={self.world_size}"
        else:
            return f"[Single-Node DDP] n_procs_per_node={self.n_procs_per_node}, world_size={self.world_size}"


@dataclass
class TrainerConfig:
    """
    # TODO: add docstring
    Args:
        n_steps (int, optional): Maximum number of steps to train for. Defaults to -1.
        n_epochs (int, optional): Maximum number of epochs to train for. Either `n_steps` or `n_epochs` must be specified. Defaults to -1.
        val_every (int, optional): Validate every `val_every` epochs. Defaults to 1.
        save_every (int, optional): Save the checkpoint every `save_every` epochs. Defaults to 0.
        save_every_idxes (list[int], optional): Save checkpoint at every index in this list. Defaults to [].
        early_stopping_patience (int, optional): Early stop training if validation metric has not improved for `early_stopping_patience` epochs. Defaults to -1.
        early_stopping_metric (int, optional): Early stopping metric name. Defaults to None. If None use the validation loss as early stopping metric.
        early_stopping_value (int, optional): Early stop training if validation metric has surpassed early_stopping_value (e.g. accuracy 1.) Defaults to None.
        early_stopping_split (str, optional): Validation split to use for early stopping. Defaults to None.
        seed (int, optional): Seed of the experiment. Defaults to 0.
        gpu_id (int, optional): The GPU id where the experiment is run. Defaults to 0.
        num_workers (int, optional): Number of workers, for e.g. for dataloader. Defaults to 0.
        lr_scheduler_step (str, optional): Whether to step the learning rate scheduler every step or every epoch. Defaults to 'step'.
        resume_training (ResumeTrainingConfig, optional): Resume training config. Contains location of checkpoint to resume training. Defaults to None.
    """

    optimizer: Optional[NameAndKwargs] = None
    training_setup: Optional[str] = None
    n_steps: Optional[Union[int, float]] = -1
    n_epochs: Optional[Union[int, float]] = -1
    val_every: Union[int, float] = 1
    save_every: Union[int, float] = 0
    save_every_idxes: list[Union[int, float]] = field(default_factory=list)
    log_train_step_every: Union[int, float] = 1
    early_stopping_patience: Union[int, float] = -1
    early_stopping_metric: Optional[str] = None
    early_stopping_value: Optional[float] = None
    early_stopping_split: Optional[str] = None
    num_workers: int = 0
    resume_training: Optional[ResumeTrainingConfig] = None
    training_strategy: Optional[TrainingStrategyConfig] = field(default_factory=TrainingStrategyConfig)
    lr_scheduler: Optional[NameAndKwargs] = None
    lr_scheduler_step: Optional[str] = "step"  # 'step' or 'epoch'
    gradient_accumulation_steps: int = 1
    # skip the last batches of each epoch that are not enough to fill the gradient accumulation steps:
    drop_last_gradient_accumulation_steps: bool = True
    # if drop_last_gradient_accumulation_steps is False, then adapt the gradient accumulation divisor:
    adapt_last_gradient_accumulation_step_divisor: bool = True
    gradient_clip_norm: Optional[float] = None


@dataclass
class SlurmConfig:
    """The current SLURM setup only allows for training on all GPUs of a single node. More finegrained submissions e.g. on single GPUs is not supported yet.
    Args:
        account (str): Specifies the project from which the GPU budget is booked from. You may check via
                       meluxina: myquota
                       karolina: it4ifree
                       leonardo: saldo -b
        nodes (int): Number of nodes training should be distributed to.
        partition (str): Name of the queue the gob should be submitted to.
        time (str): Expected runtime of the job in the format DD:HH:MM. Lower runtime estimate has higher priority in the queue. When runtime is exceeded the job is killed.
        env_name (str): Name of the conda environment to activate when submitting to the cluster.
    """

    account: str
    nodes: int
    partition: str
    time: str
    env_name: str = ""
