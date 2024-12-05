# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import contextlib
import logging
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict
from math import isnan
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchmetrics import MetricCollection
from tqdm import tqdm

from ..config import ResumeTrainingConfig, TrainingStrategyConfig
from ..log_utils.baselogger import PREFIX_BEST_CHECKPOINT, BaseLogger
from ..log_utils.filelogger import FileLogger
from ..models.base_model import BaseModel
from ..output_loader.directories import JobDirectory
from ..run_utils.runner import Runner
from ..time_utils import Stopwatch
from ..torch_utils import get_device, set_seed
from ..utils import setup_exception_logging

LOGGER = logging.getLogger(__name__)

RUN_PROGRESS_MEASURE_STEP = "train_step"
RUN_PROGRESS_MEASURE_EPOCH = "epoch"


# TODO adapt to basetrainer_ddp
# changelog:
# - use move_to() for batch
# - use batch class for access to batch data
class BaseTrainer(Runner, ABC):
    def __init__(
        self,
        config: Any,
        experiment_dir: str,
        n_steps: int = -1,
        n_epochs: int = -1,
        val_every: int = 1,
        save_every: int = 0,
        save_every_idxes: List[int] = [],
        early_stopping_patience: int = -1,
        early_stopping_metric: Optional[str] = None,
        early_stopping_value: Optional[float] = None,
        seed: int = 0,
        gpu_id: int = 0,
        num_workers: int = 0,
        lr_scheduler_step: str = "step",
        training_strategy: TrainingStrategyConfig = TrainingStrategyConfig(),
        resume_training: ResumeTrainingConfig = None,
        logger_class: Type[BaseLogger] = FileLogger,
    ):
        """Base class for all pytorch supervised-trainers. Takes care of early stopping and checkpointing.

        Args:
            experiment_dir (str): The directory where all training data is stored.
            n_steps (int, optional): Maximum number of steps to train for. Defaults to -1.
            n_epochs (int, optional): Maximum number of epochs to train for. Either `n_steps` or `n_epochs` must be specified. Defaults to -1.
            val_every (int, optional): Validate every `val_every` epochs. Defaults to 1.
            save_every (int, optional): Save the checkpoint every `save_every` epochs. Defaults to 0.
            save_every_idxes (List[int], optional): Save checkpoint at every index in this list. Defaults to [].
            early_stopping_patience (int, optional): Early stop training if validation metric has not improved for `early_stopping_patience` epochs. Defaults to -1.
            early_stopping_metric (str, optional): Metric that is used for early stopping
            seed (int, optional): Seed of the experiment. Defaults to 0.
            gpu_id (int, optional): The GPU id where the experiment is run. Defaults to 0.
            num_workers (int, optional): Number of workers, for e.g. for dataloader. Defaults to 0.
            lr_scheduler_step (str, optional): Whether to step the learning rate scheduler every step or every epoch. Defaults to 'step'.
            resume_training (ResumeTrainingConfig, optional): Resume training config. Contains location of checkpoint to resume training. Defaults to None.
        """
        super().__init__(runner_dir=experiment_dir)
        # parameters
        self.config = config
        self._experiment_dir = experiment_dir
        self._seed = int(seed)
        self._gpu_id = int(gpu_id)
        self._num_workers = int(num_workers)
        self.device = get_device(self._gpu_id)
        self._resume_training = resume_training
        self._training_strategy = training_strategy
        self._main_validation_metric = early_stopping_metric

        self._n_epochs = int(n_epochs)
        self._n_steps = int(n_steps)
        assert (self._n_steps >= 0 and not self._n_epochs >= 0) or (
            not self._n_steps >= 0 and self._n_epochs >= 0
        ), "Must either specify maximum number of epochs or maximum number of steps, but not both."
        self._val_every = int(val_every)
        self._save_every = int(save_every)
        self._save_every_idxes = save_every_idxes
        self._early_stopping_patience = int(early_stopping_patience)
        self._early_stopping_value = early_stopping_value
        self._lr_scheduler_step = lr_scheduler_step
        assert self._lr_scheduler_step in ["step", "epoch"], 'lr_scheduler_step must be either "step" or "epoch".'

        # member variables
        self._datasets = None
        self._loaders = None
        self._model: BaseModel = None
        self._optimizer: optim.Optimizer = None
        self._lr_scheduler: lr_scheduler._LRScheduler = None
        self._loss: nn.Module = None
        self._train_metrics: MetricCollection = None
        self._val_metrics: MetricCollection = None
        self._logger: BaseLogger = logger_class(
            log_dir=self._experiment_dir, config=asdict(self.config), experiment_data=self.config.experiment_data
        )

        enable_gradscaler = (
            self._training_strategy.enable_mixed_precision
            and self._training_strategy.enable_autocast_gradscaler
            and self._training_strategy.precision == "float16"
        )
        self._gradscaler = torch.cuda.amp.GradScaler(enabled=enable_gradscaler)

        # progress variables
        self._best_model_checkpoint_data: Dict[str, Any] = None
        self._progress_measure = RUN_PROGRESS_MEASURE_EPOCH if self._n_epochs > 0 else RUN_PROGRESS_MEASURE_STEP
        self._train_step_idx = 0
        self._epoch_idx = 0
        self._best_idx = 0
        self._best_val_score: float = None

        set_seed(self._seed)
        setup_exception_logging()
        LOGGER.info(f"Logging experiment data to directory: {self._experiment_dir}.")

    def _initialize(self):
        self._logger.setup_logger()
        self._create_datasets()
        self._create_dataloaders()
        self._create_model()
        # log number of parameters
        self._logger.log_keys_vals(
            prefix="num_params", keys_val={"num_model_params": self._model.num_parameters}, log_to_console=True
        )
        self._create_loss()
        self._create_metrics()

        self._model.to(device=self.device)
        if self._training_strategy.use_torch_compile:
            LOGGER.info("Compiling model...")
            with Stopwatch() as sw:
                self._model = torch.compile(self._model, **self._training_strategy.torch_compile_kwargs)
            LOGGER.info(f"Model compiled in {sw.elapsed_minutes} mins.")

        if hasattr(self._loss, "to"):
            self._loss.to(device=self.device)
        if self._train_metrics is not None:
            self._train_metrics.to(device=self.device)
        if self._val_metrics is not None:
            self._val_metrics.to(device=self.device)

        self._create_optimizer_and_scheduler(self._model)
        self._train_step_idx = 0
        self._epoch_idx = 0

    def run(self) -> None:
        self.train()

    @property
    @abstractmethod
    def config_dict(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _create_datasets(self) -> None:
        pass

    @property
    def main_validation_metric(self):
        return (
            next(iter(self._val_metrics.items()))[0]
            if self._main_validation_metric is None
            else self._main_validation_metric
        )

    @abstractmethod
    def _create_dataloaders(self) -> None:
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

    def _train_epoch(self, epoch: int) -> None:
        # setup logging
        losses_epoch: List[Dict[str, torch.Tensor]] = []

        # training loop (iterations per epoch)
        pbar = tqdm(self._loaders["train"], desc=f"Train epoch {epoch}", file=sys.stdout)
        for batch_idx, batch in enumerate(pbar):
            self._model.train()
            with Stopwatch() as sw:
                loss_dict = self._train_step(train_batch=batch, batch_idx=batch_idx)
            if loss_dict:
                if self._train_step_idx % 10 == 0:  # do not log T_train_step every step #TODO make this configurable
                    time_dict = {"T_train_step": sw.elapsed_seconds}
                    # calculate tokens per second
                    n_tokens = loss_dict.pop("n_tokens", None)
                    if n_tokens is not None:
                        time_dict["tokens_per_second"] = n_tokens / sw.elapsed_seconds
                    self._logger.log_keys_vals(
                        prefix="timer", epoch=self._epoch_idx, train_step=self._train_step_idx, keys_val=time_dict
                    )

                self._train_step_idx += 1
                losses_epoch.append(loss_dict)

                if self._lr_scheduler is not None and self._lr_scheduler_step == "step":
                    # step lr scheduler on train step
                    self._lr_scheduler.step()

                # Training termination condition
                if self._progress_measure == RUN_PROGRESS_MEASURE_STEP:
                    if self._validation_and_early_stopping(idx=self._train_step_idx, specifier=self._progress_measure):
                        break
                    elif self._train_step_idx >= (self._n_steps):
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
    def _train_step(self, train_batch, batch_idx: int) -> Dict[str, Union[float, torch.Tensor]]:
        return {}

    def _val_epoch(self, progress_idx: int, trained_model: nn.Module) -> float:
        """Implementation of one validation epoch.

        Args:
            progress_idx (int): Epoch or step index.
            trained_model (nn.Module): Model to validate

        Returns:
            float: Metric value used for early stopping
        """
        losses_epoch: List[Dict[str, torch.Tensor]] = []

        val_loader = self._loaders.get("val", None)
        if val_loader is None or self._val_metrics is None:
            return None

        pbar = tqdm(val_loader, desc=f"Val after {self._progress_measure} {progress_idx}", file=sys.stdout)
        with torch.no_grad():
            for xs, ys in pbar:
                xs, ys = xs.to(self.device), ys.to(self.device)

                with self._get_amp_context():
                    y_pred = trained_model(xs)
                # val metrics contain the training loss
                m_val = self._val_metrics(y_pred.float(), ys)

        # compute mean metrics over dataset
        metrics_epoch = self._val_metrics.compute()
        self._logger.log_keys_vals(
            prefix="val",
            epoch=self._epoch_idx,
            train_step=self._train_step_idx,
            keys_val=metrics_epoch,
            log_to_console=True,
        )
        val_score = metrics_epoch[self.main_validation_metric].item()
        self._reset_metrics("val")
        self._hook_on_val_epoch_end(progress_idx=progress_idx, trained_model=trained_model)
        return val_score

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

    def _create_checkpoint(self) -> None:
        checkpoint = {}
        # model
        if isinstance(self._model, BaseModel):
            checkpoint.update(self._model.get_checkpoint_data())
        else:
            checkpoint["model_state_dict"] = self._model.state_dict()
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
        checkpoint["__job_directory"] = str(self._experiment_dir)
        # config
        checkpoint["__config"] = self.config_dict

        idx = self._train_step_idx if self._progress_measure == RUN_PROGRESS_MEASURE_STEP else self._epoch_idx
        self._logger.save_checkpoint(checkpoint, idx=idx, specifier=self._progress_measure)

    def _resume_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # model
        self._model.load_state_dict(checkpoint["model_state_dict"])
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

    def train(self) -> Dict[str, Any]:
        """Train for n_epochs using early-stopping, epoch counter starts with 1.

        Returns:
            Dict[str, Any]: the final results
        """
        self._hook_before_initialization()
        self._initialize()
        if self._resume_training:
            LOGGER.info(f"Resume from checkpoint: {str(self._resume_training)}")
            checkpoint = JobDirectory.load_resume_checkpoint(**asdict(self._resume_training), device=self.device)
            self._resume_from_checkpoint(checkpoint=checkpoint)

        self._hook_on_training_start()
        if self._progress_measure == RUN_PROGRESS_MEASURE_STEP:
            # no number of epochs given
            # calculate from number of steps
            num_steps_per_epoch = len(self._loaders["train"])
            self._n_epochs = math.ceil(self._n_steps / num_steps_per_epoch)

        LOGGER.info(f"Starting training with progress measure `{self._progress_measure}`.")
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
        self._best_model_checkpoint_data = self._model.get_checkpoint_data()
        self._train_step_idx += 1
        self._epoch_idx += 1
        # Notes on epoch counting:
        # - epoch and step counter start with 1
        # - initial validation is done with untrained model and has epoch_idx and step_idx 0
        try:
            # Main training loop, it is in a try-except block to ensure that the training is stopped gracefully
            # when an exception is raised (e.g. checkpointing, logging, etc.).
            with Stopwatch() as sw_total:  # we also measure the total training time
                while True:
                    self._model.train()
                    with Stopwatch() as sw:
                        self._train_epoch(self._epoch_idx)
                    self._logger.log_keys_vals(
                        prefix="timer",
                        epoch=self._epoch_idx,
                        train_step=self._train_step_idx,
                        keys_val={"T_train_epoch": sw.elapsed_seconds},
                    )

                    # Training termination condition
                    if self._progress_measure == RUN_PROGRESS_MEASURE_EPOCH and self._validation_and_early_stopping(
                        idx=self._epoch_idx, specifier=self._progress_measure
                    ):
                        break
                    elif self._epoch_idx >= self._n_epochs:
                        break
                    else:
                        self._epoch_idx += 1

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
                # save best model, if it is not the initial model
                if self._best_idx > 0:
                    self._logger.save_checkpoint(
                        self._best_model_checkpoint_data,
                        idx=self._best_idx,
                        specifier=self._progress_measure,
                        name="model",
                    )

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
        model_saved = False
        if (self._save_every > 0 and idx % self._save_every == 0) or idx in self._save_every_idxes:
            self._create_checkpoint()
            model_saved = True

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
                self._best_model_checkpoint_data = self._model.get_checkpoint_data()

            if self._early_stopping_value is not None:
                if (lower_is_better and val_score <= self._early_stopping_value) or (
                    not lower_is_better and val_score >= self._early_stopping_value
                ):
                    LOGGER.info(
                        f"Early Stopping Value reached: {val_score} {'<=' if lower_is_better else '>='} {self._early_stopping_value}"
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
