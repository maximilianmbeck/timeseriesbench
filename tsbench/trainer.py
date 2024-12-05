# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import logging
from typing import Any, Union

import torch
from torchmetrics import MetricCollection

from .config import Config
from .metrics import get_metrics
from .ml_utils.data.datasetgeneratorinterface import (
    DatasetGeneratorInterface,
    DatasetGeneratorWrapper,
)
from .ml_utils.models.base_model import BaseModel
from .ml_utils.torch_utils.losses import create_loss
from .ml_utils.torch_utils.optimizer_scheduler import (
    create_optimizer_and_scheduler_from_config,
)
from .ml_utils.trainer.basetrainer_ddp import BaseTrainer
from .ml_utils.trainer.train_utils import DataBatch, ValueAccumulator, unwrap_model

LOGGER = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    config_class = Config

    def __init__(self, config: Config, model: BaseModel, datasetgenerator: DatasetGeneratorInterface):
        super().__init__(config=config)
        self.config = config
        assert (
            self.config.trainer.gradient_clip_norm is None or self.config.trainer.gradient_clip_norm > 0
        ), "gradient_clip_val must be None or > 0"
        self._log_train_step_every = self.config.trainer.log_train_step_every
        self._gradient_accumulation_steps = self.config.trainer.gradient_accumulation_steps
        self._gradient_accumulation_batch_counter = 0

        self._datasetgenerator = DatasetGeneratorWrapper(
            datasetgenerator=datasetgenerator,
            stateful_train_dataset=config.data.stateful_train_dataset,
            global_batch_size=config.global_batch_size,
            seed=config.experiment_data.seed,
            limit_n_train_samples=config.data.limit_n_train_samples,
            limit_n_val_samples=config.data.limit_n_val_samples,
        )
        self._model = model
        self._train_step_metrics_accumulator = ValueAccumulator()

    def _train_step(self, train_batch, batch_idx: int, num_batches: int) -> dict[str, Union[float, torch.Tensor]]:
        batch = DataBatch(train_batch).move_to(self.device)

        # skip last gradient accumulation steps if not enough remaining to fill gradient accumulation steps
        gradient_accumulation_divisor = self._gradient_accumulation_steps
        if (
            self.config.trainer.drop_last_gradient_accumulation_steps
            and num_batches % self._gradient_accumulation_steps != 0
        ):
            if (batch_idx + 1) > num_batches - num_batches % self._gradient_accumulation_steps:
                # skip batch
                self._gradient_accumulation_batch_counter = 0
                LOGGER.info(
                    f"[GA batch {self._gradient_accumulation_batch_counter}] Skipping batch {batch_idx} (not enough batches remaining to fill gradient accumulation steps)"
                )
                return {}
        else:
            # if not skipping batches, adapt gradient accumulation divisor
            if (
                self.config.trainer.adapt_last_gradient_accumulation_step_divisor
                and (batch_idx + 1) > num_batches - num_batches % self._gradient_accumulation_steps
            ):
                gradient_accumulation_divisor = num_batches % self._gradient_accumulation_steps

        # forward pass
        with self._get_amp_context():
            ys_pred = self._model(batch.xs, **batch.meta)
            loss = self._loss(ys_pred, batch.ys)
            ga_scaled_loss = loss / gradient_accumulation_divisor

        LOGGER.debug(f"[GA batch {self._gradient_accumulation_batch_counter}] Loss: {loss.item()}")

        # backward pass
        self._gradscaler.scale(ga_scaled_loss).backward()
        self._gradient_accumulation_batch_counter += 1

        # metrics
        if self._train_metrics is None:
            metric_vals = {}
        else:
            with torch.no_grad():
                metric_vals = self._train_metrics(ys_pred.float(), batch.ys)

        # accumulate loss & metrics
        self._train_step_metrics_accumulator.add({"loss": loss, **metric_vals})

        # gradient accumulation
        if (batch_idx + 1) % self._gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
            # Do train step with accumulated gradients
            # reduce & reset accumulated metrics
            loss_dict = self._train_step_metrics_accumulator.reduce(reduce_fn="mean", dist_sync_before_reduce=True)
            self._train_step_metrics_accumulator.reset()

            # gradient clipping
            if self.config.trainer.gradient_clip_norm is not None:
                self._gradscaler.unscale_(self._optimizer)

                if hasattr(self._model, "clip_grad_norm_"):
                    # We need a special case for FSDP since the gradients are sharded. FSDP fortunately provides
                    # the corresponding function for us.
                    grad_norm = self._model.clip_grad_norm_(self.config.trainer.gradient_clip_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), self.config.trainer.gradient_clip_norm
                    )
                loss_dict["grad_norm"] = grad_norm

            self._gradscaler.step(self._optimizer)
            self._gradscaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self._optimizer.zero_grad(set_to_none=True)
            self._gradient_accumulation_batch_counter = 0

            # log learning rate, assume only one parameter group
            loss_dict["lr"] = self._optimizer.param_groups[0]["lr"]
            # log step
            self._log_step(log_dict=loss_dict)
        else:
            # No train step, accumulate current batch gradient
            loss_dict = {}  # empty dict indicates gradient accumulation step

        return loss_dict

    def _create_dataloaders(self) -> None:
        # unused
        pass

    def _create_datasets(self) -> None:
        self._train_dataset = self._datasetgenerator.train_split
        self._val_datasets = self._datasetgenerator.validation_split
        self._collate_fn = self._datasetgenerator.collate_fn

    def _create_loss(self) -> None:
        LOGGER.info("Creating loss.")
        self._loss = create_loss(self.config.loss)

    def _create_metrics(self) -> None:
        from .ml_utils.torch_utils.metrics import Loss

        LOGGER.info("Creating metrics.")
        self._train_metrics, self._val_metrics = get_metrics(self.config.metrics, self._datasetgenerator)
        if self._val_metrics is None:
            self._val_metrics = MetricCollection(metrics=[Loss(self._loss)])
        else:
            self._val_metrics = MetricCollection(metrics=[Loss(self._loss), self._val_metrics])

    def _create_model(self) -> None:
        self._logger.watch_model(self._model)
        LOGGER.info(f"Model trained: \n{self._model}")
        LOGGER.info(f"Parameter Table: \n{self._model.parameter_table}")

    def _create_optimizer_and_scheduler(self, model, *args, **kwargs) -> None:
        # We need to initialize the optimzer after we wrap the model in FSDP.
        # However, DDP wraps slightly different and it does not give access to all members
        # of the underlying module. We therefore need to expose the optim
        # groups of the underlying model explicitly here.

        model = unwrap_model(model)

        LOGGER.info("Creating optimizer and scheduler.")
        if hasattr(model, "configure_optim_groups"):
            optim_groups = model.configure_optim_groups(self.config.trainer.optimizer.kwargs)
        else:
            LOGGER.warning("No optim groups found. Using all parameters.")
            optim_groups = [{"params": model.parameters()}]

        self._optimizer, self._lr_scheduler = create_optimizer_and_scheduler_from_config(
            optim_groups,
            optimizer_cfg=self.config.trainer.optimizer,
            lr_scheduler_cfg=self.config.trainer.lr_scheduler,
        )

    def _log_step(
        self,
        log_dict: dict[str, torch.Tensor],
        additional_logs_step: dict[str, Any] = {},
    ) -> None:
        if self._train_step_idx % self._log_train_step_every == 0:
            log_dict = {**log_dict, **additional_logs_step}
            self._logger.log_keys_vals(
                prefix="train_step",
                train_step=self._train_step_idx + 1,  # +1 because we log after the step #TODO check this
                epoch=self._epoch_idx,
                keys_val=log_dict,
                log_to_console=False,
            )
