# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import contextlib
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict
from math import isnan
from typing import Any, Callable, Mapping, Optional, Union

import pandas as pd
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MetricCollection

from ...config import Config
from ..data.dataloaders import create_train_loader, create_val_loaders
from ..ddp_utils.setup import is_distributed, is_rank0
from ..log_utils.baselogger import PREFIX_BEST_CHECKPOINT, BaseLogger
from ..log_utils.filelogger import FileLogger
from ..log_utils.log_cmd import get_tqdm_pbar
from ..log_utils.nooplogger import NoOpLogger
from ..log_utils.wandblogger import WandBLogger
from ..models.base_model import BaseModel
from ..output_loader.directories import JobDirectory
from ..run_utils.runner import Runner
from ..time_utils import Stopwatch
from ..torch_utils import get_device
from ..utils import call_on_wrapped_object
from .train_utils import DataBatch, unwrap_model

LOGGER = logging.getLogger(__name__)

RUN_PROGRESS_MEASURE_STEP = "train_step"
RUN_PROGRESS_MEASURE_EPOCH = "epoch"
VAL_SPLIT_PREFIX = "val"
VAL_SPLIT_NAME_TEMPLATE = "{prefix}_{split_name}"
MAX_N_EPOCHS = 1000000

# TODOs major refactor:
# - reimplement the training loops in a more modular way,
#   i.e. add functions that are called in the respective methods (then the functions are easily testable)
# TODOs minor refactor:


class BaseTrainer(Runner, ABC):
    def __init__(self, config: Config):
        """Base class for all pytorch supervised-trainers. Takes care of early stopping and checkpointing."""
        super().__init__(runner_dir=config.experiment_data.experiment_dir)
        # parameters
        self.config = config
        self.trainer_cfg = config.trainer
        LOGGER.info(f"Logging experiment data to directory: {config.experiment_data.experiment_dir}.")
        self._seed = int(config.experiment_data.seed)
        self._gpu_id = int(config.experiment_data.gpu_id)
        self.device = get_device(self._gpu_id)
        self._resume_training = self.trainer_cfg.resume_training
        self._training_strategy = self.trainer_cfg.training_strategy
        self._ddp_config = config.ddp

        self._n_epochs = int(self.trainer_cfg.n_epochs)
        self._n_steps = int(self.trainer_cfg.n_steps)
        assert (self._n_steps >= 0 and not self._n_epochs >= 0) or (
            not self._n_steps >= 0 and self._n_epochs >= 0
        ), "Must either specify maximum number of epochs or maximum number of steps, but not both."
        self._val_every = int(self.trainer_cfg.val_every)
        self._save_every = int(self.trainer_cfg.save_every)
        self._early_stopping_patience = int(self.trainer_cfg.early_stopping_patience)
        self._lr_scheduler_step = self.trainer_cfg.lr_scheduler_step
        assert self._lr_scheduler_step in ["step", "epoch"], 'lr_scheduler_step must be either "step" or "epoch".'

        # member variables
        # datasets
        self._train_dataset: Dataset = None
        self._val_datasets: Optional[Mapping[str, Dataset]] = None
        self._collate_fn: Callable = None
        # will be determined on initialization, used for creating dataloaders for stateful datasets when resuming training
        self._num_training_samples: int = None
        # training variables
        self._model: BaseModel = None
        self._optimizer: optim.Optimizer = None
        self._lr_scheduler: lr_scheduler._LRScheduler = None
        self._loss: Union[nn.Module, Callable] = None
        self._train_metrics: MetricCollection = None
        self._val_metrics: MetricCollection = None
        if self._is_ddp() and not is_rank0():
            # only rank zero process should log
            logger_class = NoOpLogger
        else:
            logger_class = FileLogger if config.wandb is False else WandBLogger
        self._logger: BaseLogger = logger_class(
            log_dir=self.config.experiment_data.experiment_dir,
            config=asdict(self.config),
            experiment_data=self.config.experiment_data,
        )

        enable_gradscaler = (
            self._training_strategy.enable_mixed_precision
            and self._training_strategy.enable_autocast_gradscaler
            and self._training_strategy.precision == "float16"
        )
        self._gradscaler = torch.cuda.amp.GradScaler(enabled=enable_gradscaler)

        # progress variables
        self._progress_measure = RUN_PROGRESS_MEASURE_EPOCH if self._n_epochs > 0 else RUN_PROGRESS_MEASURE_STEP
        self._train_step_idx = 0
        self._epoch_idx = 0
        self._best_idx = 0
        self._best_val_score: float = None
        self._training_done: bool = False
        # will be set when resuming training
        self._resume_batch_idx: int = None

    def _initialize(self):
        self._logger.setup_logger()
        self._create_datasets()
        self._num_training_samples = len(self._train_dataset)
        self._create_model()
        # log number of parameters
        self._logger.log_keys_vals(
            prefix="num_params", keys_val={"num_model_params": self._model.num_parameters}, log_to_console=True
        )
        self._create_loss()
        self._create_metrics()
        train_dtype = self._get_train_dtype()
        LOGGER.info(f"Training device: {self.device}")
        LOGGER.info(f"Training dtype: {train_dtype}")
        self._model.to(device=self.device, dtype=train_dtype)

        if hasattr(self._loss, "to"):
            self._loss.to(device=self.device)
        if self._train_metrics is not None:
            self._train_metrics.to(device=self.device)
        if self._val_metrics is not None:
            self._val_metrics.to(device=self.device)

        self._create_optimizer_and_scheduler(self._model)
        self._ddp_setup()
        # compile model after it is wrapped with DDP
        # see: https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860
        if self._training_strategy.use_torch_compile:
            LOGGER.info("Compiling model...")
            with Stopwatch() as sw:
                self._model = torch.compile(self._model, **self._training_strategy.torch_compile_kwargs)
            LOGGER.info(f"Model compiled in {sw.elapsed_minutes} mins.")
        self._train_step_idx = 0
        self._epoch_idx = 0

    def run(self) -> None:
        self.train()

    @property
    def num_train_samples(self) -> int:
        """Returns the number of training samples."""
        return self._num_training_samples

    @property
    def num_train_batches(self) -> int:
        """Returns the number of training batches."""
        from math import ceil

        if self.config.data.dl_kwargs.drop_last:
            num_batches = self.num_train_samples // self.config.global_batch_size
        else:
            num_batches = ceil(self.num_train_samples / self.config.global_batch_size)
        return num_batches

    @property
    def main_validation_metric(self):
        return (
            next(iter(self._val_metrics.items()))[0]
            if self.trainer_cfg.early_stopping_metric is None
            else self.trainer_cfg.early_stopping_metric
        )

    @abstractmethod
    def _create_datasets(self) -> None:
        pass

    @abstractmethod
    def _create_model(self) -> None:
        pass

    @abstractmethod
    def _create_optimizer_and_scheduler(self, model: nn.Module) -> None:
        pass

    @abstractmethod
    def _create_loss(self) -> None:
        """Create loss for optimization."""
        pass

    @abstractmethod
    def _create_metrics(self) -> None:
        """Create a list of metrics for training and validation.
        The first entry in val_metric is used for early stopping.
        """
        pass

    def _reset_metrics(self, which: str = "all") -> None:
        if self._train_metrics is not None and (which == "all" or which == "train"):
            self._train_metrics.reset()
        if self._val_metrics is not None and (which == "all" or which == "val"):
            self._val_metrics.reset()

    def _create_train_dataloader(self, epoch: int) -> DataLoader:
        assert self._train_dataset is not None, "train_dataset must not be None."

        if self._resume_batch_idx is not None:
            # resume training from checkpoint
            batch_idx = self._resume_batch_idx
            # reset resume batch index so that next epoch starts with batch index 0
            self._resume_batch_idx = None
        else:
            batch_idx = 0
        call_on_wrapped_object(
            wrapper=self._train_dataset,
            wrapped_obj_name="dataset",
            method="set_epoch",
            kwargs=dict(epoch=epoch, batch_idx=batch_idx),
            error_when_not_called=False,
        )
        train_dl = create_train_loader(train_split=self._train_dataset, collate_fn=self._collate_fn, config=self.config)

        if self._is_ddp():
            train_dl.sampler.set_epoch(epoch)

        return train_dl

    def _train_epoch(self, epoch: int) -> None:
        train_dataloader = self._create_train_dataloader(epoch=epoch)

        # setup logging
        losses_epoch: list[dict[str, torch.Tensor]] = []

        # training loop (iterations per epoch)

        pbar = get_tqdm_pbar(train_dataloader, desc=f"Train epoch {epoch}", file=sys.stdout)
        for batch_idx, batch in enumerate(pbar):
            LOGGER.debug(f"Train Epoch: {epoch} Batch: {batch_idx}")
            self._model.train()
            with Stopwatch() as sw:
                loss_dict = self._train_step(train_batch=batch, batch_idx=batch_idx, num_batches=len(train_dataloader))
            if bool(loss_dict):  # if loss_dict not empty
                # increase step counter before step
                self._train_step_idx += 1
                if self._train_step_idx % 10 == 0:  # do not log T_train_step every step #TODO make this configurable
                    time_dict = {"T_train_step": sw.elapsed_seconds}
                    # calculate tokens per second
                    n_tokens = loss_dict.get("n_tokens", None)
                    if n_tokens is not None:
                        time_dict["tokens_per_second"] = n_tokens / sw.elapsed_seconds
                    self._logger.log_keys_vals(
                        prefix="timer", epoch=self._epoch_idx, train_step=self._train_step_idx, keys_val=time_dict
                    )

                losses_epoch.append(loss_dict)

                if self._lr_scheduler is not None and self._lr_scheduler_step == "step":
                    # step lr scheduler on train step
                    self._lr_scheduler.step()

                # Training termination condition
                if self._progress_measure == RUN_PROGRESS_MEASURE_STEP:
                    if self._validation_and_early_stopping(idx=self._train_step_idx, specifier=self._progress_measure):
                        self._training_done = True
                        break
                    elif self._train_step_idx >= (self._n_steps):
                        LOGGER.info(f"Reached max number of steps: {self._n_steps}")
                        self._training_done = True
                        break

        if self._lr_scheduler is not None and self._lr_scheduler_step == "epoch":
            # step lr scheduler on train epoch
            self._lr_scheduler.step()

        # log epoch
        if self._train_metrics is not None:
            metrics_epoch = self._train_metrics.compute()
        else:
            metrics_epoch = {}
        self._logger.log_keys_vals(
            prefix="train",
            epoch=self._epoch_idx,
            train_step=self._train_step_idx,
            keys_multiple_vals=losses_epoch,
            keys_val=metrics_epoch,
            log_to_console=True,
        )

        self._reset_metrics("train")

    @abstractmethod
    def _train_step(self, train_batch, batch_idx: int, num_batches: int) -> dict[str, Union[float, torch.Tensor]]:
        return {}

    def _create_val_dataloader(self) -> Optional[Mapping[str, DataLoader]]:
        if self._val_datasets is None:
            return None

        val_dataloader = create_val_loaders(
            val_splits=self._val_datasets, collate_fn=self._collate_fn, config=self.config
        )
        return val_dataloader

    def _val_epoch(self, progress_idx: int, trained_model: nn.Module) -> float:
        """Do one validation epoch for all val splits."""

        val_dataloader = self._create_val_dataloader()

        if val_dataloader is None or self._val_metrics is None or len(val_dataloader) == 0:
            return None

        val_scores = {}
        for val_split_name, val_split_loader in val_dataloader.items():
            if len(val_dataloader) == 1:
                prefix_name = VAL_SPLIT_PREFIX
            else:
                prefix_name = VAL_SPLIT_NAME_TEMPLATE.format(prefix=VAL_SPLIT_PREFIX, split_name=val_split_name)
            val_score = self._val_epoch_for_single_val_split(
                progress_idx=progress_idx,
                trained_model=trained_model,
                val_split_name=prefix_name,
                val_split_loader=val_split_loader,
            )
            val_scores[val_split_name] = val_score

        if self.trainer_cfg.early_stopping_split is not None:
            val_score = val_scores[self.trainer_cfg.early_stopping_split]
        else:
            # get the first validation score if early stopping split is not explicitely configured
            # note dictionary are insertion ordered from Python 3.7 on:
            # https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6
            val_score = next(iter(val_scores.values()))

        self._hook_on_val_epoch_end(progress_idx=progress_idx, trained_model=trained_model)
        return val_score

    def _val_epoch_for_single_val_split(
        self, progress_idx: int, trained_model: nn.Module, val_split_name: str, val_split_loader: DataLoader
    ) -> float:
        """Implementation of one validation epoch.

        Args:
            progress_idx (int): Epoch or step index.
            trained_model (nn.Module): Model to validate

        Returns:
            float: Metric value used for early stopping
        """
        self._reset_metrics("val")

        pbar = get_tqdm_pbar(
            val_split_loader, desc=f"Val after {self._progress_measure} {progress_idx}", file=sys.stdout
        )
        with torch.no_grad():
            for batch in pbar:
                self._val_step(trained_model=trained_model, val_batch=batch)

        # compute mean metrics over dataset
        metrics_epoch = self._val_metrics.compute()
        self._logger.log_keys_vals(
            prefix=val_split_name,
            epoch=self._epoch_idx,
            train_step=self._train_step_idx,
            keys_val=metrics_epoch,
            log_to_console=True,
        )
        val_score = metrics_epoch[self.main_validation_metric].item()
        self._reset_metrics("val")
        return val_score

    def _val_step(self, trained_model: nn.Module, val_batch) -> None:
        """A single validation step. Used for validation during training.
        This does not return any metrics, but updates the validation metrics.
        We are only interested in the aggregated metrics over all samples, which we compute in _val_epoch().
        """
        batch = DataBatch(val_batch).move_to(self.device)
        with self._get_amp_context():
            y_pred = trained_model(batch.xs, **batch.meta)
            # val metrics contain the training loss
            self._val_metrics.update(y_pred.float(), batch.ys)

    def _val_lower_is_better(self) -> bool:
        """Return the value for the first validation metric in the metric collection."""
        # index 1 is the Metrics class
        return not self._val_metrics[self.main_validation_metric].higher_is_better

    def _hook_before_initialization(self, *args, **kwargs) -> None:
        pass

    def _hook_on_training_start(self, *args, **kwargs) -> None:
        pass

    def _hook_on_training_end(self, *args, **kwargs) -> None:
        pass

    def _hook_on_val_epoch_end(self, *args, **kwargs) -> None:
        pass

    def _create_checkpoint(self, save_to_disk=True) -> dict[str, Any]:
        """Create a checkpoint of the current training state."""
        checkpoint = {}
        # NOTE: We always store the model state dict, even if it is in DDP mode
        # model
        temp_model = unwrap_model(self._model)
        if isinstance(temp_model, BaseModel):
            checkpoint.update(temp_model.get_checkpoint_data())
        else:
            checkpoint["model_state_dict"] = temp_model.state_dict()
        # optimizer
        checkpoint["optimizer_state_dict"] = self._optimizer.state_dict()
        # grad scaler
        if self._gradscaler is not None:
            checkpoint["gradscaler_state_dict"] = self._gradscaler.state_dict()
        # scheduler
        if self._lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self._lr_scheduler.state_dict()
        # trainer
        trainer_data = {"train_step_idx": self._train_step_idx, "epoch_idx": self._epoch_idx}
        checkpoint["trainer_data"] = trainer_data
        # job_dir
        checkpoint["__job_directory"] = str(self.config.experiment_data.experiment_dir)
        # config
        checkpoint["__config"] = asdict(self.config)
        # save
        if save_to_disk and is_rank0():
            idx = self._train_step_idx if self._progress_measure == RUN_PROGRESS_MEASURE_STEP else self._epoch_idx
            self._logger.save_checkpoint(checkpoint, idx=idx, specifier=self._progress_measure)
        return checkpoint

    def _resume_from_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Resume training from a checkpoint."""
        # model
        temp_model = unwrap_model(self._model)
        temp_model.load_state_dict(checkpoint["model_state_dict"])
        # optimizer
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # grad scaler
        if self._gradscaler is not None:
            self._gradscaler.load_state_dict(checkpoint["gradscaler_state_dict"])
        # scheduler
        if self._lr_scheduler is not None:
            self._lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        # trainer
        trainer_data = checkpoint["trainer_data"]
        self._train_step_idx = trainer_data["train_step_idx"]
        self._epoch_idx = trainer_data["epoch_idx"]

    def _get_amp_context(self) -> contextlib.AbstractContextManager:
        if self._training_strategy.enable_mixed_precision:
            # infer device type from gpu_id
            device_type = "cuda" if self._gpu_id >= 0 else "cpu"
            return torch.autocast(
                device_type=device_type,
                dtype=self._training_strategy.precision_dtype,
                enabled=self._training_strategy.enable_autocast_gradscaler,
            )
        else:
            return contextlib.nullcontext()

    def _get_train_dtype(self) -> torch.dtype:
        if self._training_strategy.enable_mixed_precision:
            return self._training_strategy.precision_dtype
        else:
            return torch.float32

    def _ddp_setup(self) -> None:
        """Setup DDP training.
        Wrap model and perform checks on distributed sampler"""
        if self._is_ddp():
            LOGGER.info("Wrapping model for DDP training..")
            self._model = nn.parallel.DistributedDataParallel(self._model, static_graph=self._ddp_config.static_graph)
            LOGGER.info(f"[DPP setup] {self._ddp_config.get_ddp_summary()}")

    def _is_ddp(self) -> bool:
        """Return true if training in data distributed parallel mode."""
        is_ddp = self._ddp_config is not None and self._ddp_config.enable_ddp
        assert is_ddp == is_distributed(), "DDP is enabled but not running in distributed mode."
        return is_ddp

    def train(self) -> dict[str, Any]:
        """Train for n_epochs using early-stopping, epoch counter starts with 1.

        The training loop has the following structure / calling order:

        - initialize: [epoch=0, train_idx=0]
            [- resume_training: [epoch=X, train_idx=Y]]
        - val_epoch # at epoch=0|X, train_idx=0|Y
        > decrease:
          if resume_training:
            if train_idx % len(train_dataloader) == 0:
                epoch -= 1 # we need to decrease epoch by 1, because we will increase it at start of epoch again
        # At start of train loop: epoch=0, train_idx=0 or epoch=X|X+1, train_idx=Y
        - train:
            > increase: epoch += 1
            - train_epoch:
                - train_step
                - - log step + 1
                > increase: train_idx += 1
                - [val_epoch, early stopping, checkpointing]
            - - log epoch
        - create final checkpoint

        Returns:
            dict[str, Any]: the final results
        """
        self._hook_before_initialization()
        self._initialize()
        if self._resume_training is not None:
            LOGGER.info(f"Resume from checkpoint: {str(self._resume_training)}")
            checkpoint = JobDirectory.load_resume_checkpoint(**asdict(self._resume_training), device=self.device)
            self._resume_from_checkpoint(checkpoint=checkpoint)

        self._hook_on_training_start()
        if self._progress_measure == RUN_PROGRESS_MEASURE_STEP:
            # no number of epochs given
            self._n_epochs = MAX_N_EPOCHS

        LOGGER.info(f"Starting training with progress measure `{self._progress_measure}`.")
        LOGGER.info(
            f"[Batch Size] global: {self.config.global_batch_size}, mini: {self.config.mini_batch_size},"
            f" gradient accumulation steps: {self.config.gradient_accumulation_steps},"
            f" number of ranks: {self.config.world_size},"
            f" number of nodes: {self.config.n_nodes}."
        )
        # log batch size and distributed setup
        self._logger.log_keys_vals(
            prefix="distributed",
            keys_val={
                "global_batch_size": self.config.global_batch_size,
                "mini_batch_size": self.config.mini_batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "num_ranks": self.config.world_size,
                "num_nodes": self.config.n_nodes,
            },
            log_to_console=True,
        )
        self._best_idx = (
            self._train_step_idx if self._progress_measure == RUN_PROGRESS_MEASURE_STEP else self._epoch_idx
        )
        self._best_val_score = float("inf") if self._val_lower_is_better() else -float("inf")

        # validate untrained model as baseline
        self._model.eval()
        with Stopwatch() as sw:  # TODO make this a decorator
            self._best_val_score = self._val_epoch(self._epoch_idx, self._model)
        self._logger.log_keys_vals(
            prefix="timer",
            epoch=self._epoch_idx,
            train_step=self._train_step_idx,
            keys_val={"T_val_epoch": sw.elapsed_seconds},
        )

        # save initialized/untrained model
        self._create_checkpoint()

        if self._resume_training is not None:
            self._resume_batch_idx = self._train_step_idx % self.num_train_batches
            if self._resume_batch_idx != 0:
                self._epoch_idx -= 1
        else:
            assert (
                self._epoch_idx == 0 and self._train_step_idx == 0
            ), "Epoch and train step index must be 0, when not resuming training."

        # Notes on epoch counting:
        # - epoch and step counter start with 0
        # - initial validation is done with untrained model and has epoch_idx and step_idx 0
        try:
            # Main training loop, it is in a try-except block to ensure that the training is stopped gracefully
            # when an exception is raised (e.g. checkpointing, logging, etc.).
            with Stopwatch() as sw_total:  # we also measure the total training time
                while True:
                    self._model.train()
                    # increase epoch counter before epoch
                    self._epoch_idx += 1
                    with Stopwatch() as sw:
                        self._train_epoch(self._epoch_idx)
                    self._logger.log_keys_vals(
                        prefix="timer",
                        epoch=self._epoch_idx,
                        train_step=self._train_step_idx,
                        keys_val={"T_train_epoch": sw.elapsed_seconds},
                    )

                    # Training termination condition
                    if self._training_done:
                        break
                    elif self._progress_measure == RUN_PROGRESS_MEASURE_EPOCH and self._validation_and_early_stopping(
                        idx=self._epoch_idx, specifier=self._progress_measure
                    ):
                        break
                    elif self._epoch_idx >= self._n_epochs:
                        LOGGER.info(f"Reached maximum number of epochs: {self._n_epochs}")
                        self._training_done = True
                        break
                    else:
                        pass

            # save checkpoint at end of training for possible resume
            self._create_checkpoint()

            final_results = {
                f"{PREFIX_BEST_CHECKPOINT}{self._progress_measure}": self._best_idx,
                f"{PREFIX_BEST_CHECKPOINT}val_score": self._best_val_score,
                "T_total_h": sw_total.elapsed_hours,
            }
            LOGGER.info(f"Final results: \n{pd.Series(final_results)}")
            LOGGER.info(f"Training time: {sw_total.elapsed_time_string}")

            if self._best_idx >= 0:
                # write best checkpoint index to file
                self._logger.save_best_checkpoint_idx(specifier=self._progress_measure, best_idx=self._best_idx)

            self._logger.finish(final_results=final_results)
            self._hook_on_training_end(final_results=final_results)

        except Exception as e:
            # no best checkpoint index is written to file
            # if training is interrupted
            # in this way this job is marked as failed
            LOGGER.error(f"Exception during training: {e}")
            LOGGER.info("Saving checkpoint before exiting...")
            self._create_checkpoint()
            self._logger.finish(exit_code=1)  # mark run as failed
            raise e

        return final_results

    def _validation_and_early_stopping(self, idx: int, specifier: str) -> bool:
        """Runs validation and early stopping.

        Args:
            idx (int): Current epoch or step index.
            specifier (str): `epoch` or `step`

        Returns:
            bool: True, if training should be early stopped.
        """
        if (self._save_every > 0 and idx % self._save_every == 0) or idx in self.trainer_cfg.save_every_idxes:
            self._create_checkpoint()

        if self._val_every > 0 and idx % self._val_every == 0:
            lower_is_better = self._val_lower_is_better()

            self._model.eval()
            with Stopwatch() as sw:
                val_score = self._val_epoch(progress_idx=idx, trained_model=self._model)
                if val_score is None:
                    return False  # keep training
            self._logger.log_keys_vals(
                prefix="timer",
                epoch=self._epoch_idx,
                train_step=self._train_step_idx,
                keys_val={"T_val_epoch": sw.elapsed_seconds},
            )
            assert isinstance(val_score, float)
            if isnan(val_score):
                raise RuntimeError(f"Validation score is NaN in {specifier} {idx}.")

            if (lower_is_better and val_score < self._best_val_score) or (
                not lower_is_better and val_score > self._best_val_score
            ):
                LOGGER.info(
                    f"New best val score: {val_score} {'<' if lower_is_better else '>'} {self._best_val_score} (old best val score)"
                )
                self._best_idx = idx
                self._best_val_score = val_score
                best_checkpoint = self._create_checkpoint(save_to_disk=False)
                best_checkpoint["__best_val_score"] = self._best_val_score
                best_checkpoint["__lower_is_better"] = lower_is_better
                if is_rank0():
                    self._logger.save_best_checkpoint(checkpoint=best_checkpoint)

            early_stopping_value = self.trainer_cfg.early_stopping_value
            if early_stopping_value is not None:
                if (lower_is_better and val_score <= early_stopping_value) or (
                    not lower_is_better and val_score >= early_stopping_value
                ):
                    LOGGER.info(
                        f"Early Stopping Value reached: {val_score} {'<=' if lower_is_better else '>='} {early_stopping_value}"
                    )
                    return True

            if self._early_stopping_patience > 0:
                if (
                    (lower_is_better and val_score >= self._best_val_score)
                    or (not lower_is_better and val_score <= self._best_val_score)
                ) and idx > self._best_idx + self._early_stopping_patience:
                    LOGGER.info(
                        "Early stopping patience exhausted. "
                        f"Best val score {self._best_val_score} in {specifier} {self._best_idx}."
                    )
                    return True
        return False
