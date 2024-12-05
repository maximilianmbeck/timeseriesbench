# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import torch
from torch import nn

import wandb

from ..config import ExperimentConfig
from ..output_loader.directories import JobDirectory
from ..utils import convert_dict_to_python_types
from .baselogger import LOG_STEP_KEY, increment_log_step_on_call
from .filelogger import FileLogger

LOGGER = logging.getLogger(__name__)


class WandBLogger(FileLogger):
    def __init__(
        self,
        log_dir: Union[str, Path, JobDirectory],
        experiment_data: ExperimentConfig,
        config: Dict[str, Any] = {},
        watch_model_args: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__(log_dir=log_dir, config=config, experiment_data=experiment_data, **kwargs)
        self._experiment_data = experiment_data
        self._config = config
        self._wandb_run = None
        self._watch_model_args = watch_model_args
        self.log_to_wandb = True

    def setup_logger(self):
        """Sets up wandb if necessary and creates directories."""
        self.logger_directory.create_directories()

        # start wandb if necessary
        if self.log_to_wandb:
            LOGGER.info("Starting wandb.")
            self._wandb_run = wandb.init(
                entity=self._experiment_data.entity,
                project=self._experiment_data.project_name,
                name=self._experiment_data.job_name,
                tags=[self._experiment_data.experiment_tag],  # TODO make tags a list
                notes=self._experiment_data.experiment_notes,
                group=self._experiment_data.experiment_tag,
                job_type=self._experiment_data.experiment_type,
                dir=self._experiment_data.output_dir,  # use default wandb dir to enable later wandb sync
                config=self._config,
                settings=wandb.Settings(start_method="fork"),
            )
        else:
            LOGGER.info("Not logging to wandb. Logging to disc only.")

    @increment_log_step_on_call
    def log_keys_vals(
        self,
        prefix: str = "default",
        epoch: int = -1,
        train_step: int = -1,
        keys_multiple_vals: Union[
            Dict[str, Union[List[torch.Tensor], torch.Tensor]], List[Dict[str, torch.Tensor]]
        ] = {},
        keys_val: Dict[str, Any] = {},
        log_to_console: bool = False,
    ) -> None:
        if keys_multiple_vals or keys_val:
            log_dict = self._create_log_dict(epoch, train_step, keys_multiple_vals, keys_val)

            log_dict_numeric_vals = convert_dict_to_python_types(log_dict)
            if log_to_console:
                # log to console
                LOGGER.info(f"{prefix} \n{pd.Series(log_dict_numeric_vals, dtype=float)}")
            else:
                LOGGER.debug(f"{prefix} \n{pd.Series(log_dict_numeric_vals, dtype=float)}")

            if self.log_to_wandb:
                wandb.log({f"{prefix}/": log_dict})

            # add the internal log_step to every log
            log_dict_numeric_vals.update({LOG_STEP_KEY: self.log_step})
            self.logs[prefix].append(log_dict_numeric_vals)

    def watch_model(self, model: nn.Module):
        if self.log_to_wandb:
            if self._watch_model_args:
                wandb.watch(model, **self._watch_model_args)

    def finish(self, final_results: Dict[str, Any] = {}, exit_code: int = 0):
        super().finish(final_results=final_results, exit_code=exit_code)
        if self.log_to_wandb:
            wandb.run.summary.update(final_results)

        if self._wandb_run:
            LOGGER.info("Finishing wandb.")
            self._wandb_run.finish(exit_code=exit_code)
