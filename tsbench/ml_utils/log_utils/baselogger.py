# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import torch
from torch import nn

from ..config import ExperimentConfig
from ..output_loader.directories import JobDirectory
from ..utils import convert_listdict_to_python_types

PREFIX_BEST_CHECKPOINT = "best_"
FN_BEST_CHECKPOINT = PREFIX_BEST_CHECKPOINT + "{specifier}.txt"
LOG_POLICY_END = "end"  # should write the files at the end

LOG_STEP_KEY = "log_step"

FN_FINAL_RESULTS = "final_results"
FN_DATA_LOG_PREFIX = "stats_"
FN_DATA_LOG = FN_DATA_LOG_PREFIX + "{datasource}.csv"

FORMAT_LOGGING = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"


def increment_log_step_on_call(method):
    """Decorator to increment the internal log step of a logging / save function of `Logger`:"""

    def inner(self, *args, **kwargs):
        method(self, *args, **kwargs)
        self.log_step += 1

    return inner


class BaseLogger(ABC):
    def __init__(
        self,
        log_dir: Union[str, Path, JobDirectory],
        config: Dict[str, Any] = {},
        experiment_data: ExperimentConfig = None,
        **kwargs,
    ):
        self.logger_directory = JobDirectory(log_dir)
        # counter for counting how often log_keys_vals was called
        self.log_step = 0
        self.logs: Dict[str, List] = defaultdict(list)

    @abstractmethod
    def setup_logger(self) -> None:
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def finish(self, final_results: Dict[str, Any] = {}, exit_code: int = 0):
        """Finishes the logging and saves the logs to disc.

        Args:
            final_results (Dict[str, Any], optional): Run summary dict. Defaults to {}.
            exit_code (int, optional): Set to something other than 0 to mark run as failed. Defaults to 0.
        """
        pass

    def watch_model(self, model: nn.Module) -> None:
        return None

    @increment_log_step_on_call
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], idx: int, specifier: str = "", name: str = "checkpoint"
    ) -> None:
        self.logger_directory.save_checkpoint(checkpoint=checkpoint, idx=idx, specifier=specifier, name=name)

    @increment_log_step_on_call
    def save_best_checkpoint(self, checkpoint: Dict[str, Any], name: str = "best_checkpoint") -> None:
        self.logger_directory.save_best_checkpoint(checkpoint=checkpoint, name=name)

    def save_best_checkpoint_idx(self, specifier: str, best_idx: int) -> None:
        """Write the best checkpoint idx to in a textfile and save it in the job directory."""
        best_checkpoint_file = self.logger_directory.checkpoints_folder / FN_BEST_CHECKPOINT.format(specifier=specifier)
        with best_checkpoint_file.open("w") as fp:
            fp.write(str(best_idx))

    def _create_log_dict(
        self,
        epoch: int = -1,
        train_step: int = -1,
        keys_multiple_vals: Union[
            Dict[str, Union[List[torch.Tensor], torch.Tensor]], List[Dict[str, torch.Tensor]]
        ] = {},
        keys_val: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """Creates a log dict from the given arguments.
        Takes the mean if multiple values are given.

        Args:
            epoch (int, optional): Epoch. Defaults to -1.
            train_step (int, optional): Train step. Defaults to -1.
            keys_multiple_vals (Union[Dict[str, Union[List[torch.Tensor], torch.Tensor]], List[Dict[str, torch.Tensor]]], optional): Multiple values. Defaults to {}.
            keys_val (Dict[str, Any], optional): Single values. Defaults to {}.

        Returns:
            Dict[str, Any]: Log dict.
        """

        # log mean value if multiple values are given
        keys_multiple_vals_df = pd.DataFrame(convert_listdict_to_python_types(keys_multiple_vals))
        keys_multiple_vals_mean = keys_multiple_vals_df.mean(axis=0).to_dict()

        log_dict = {**keys_multiple_vals_mean, **keys_val}
        if epoch > -1:
            log_dict.update({"epoch": epoch})
        if train_step > -1:
            log_dict.update({"train_step": train_step})
        return log_dict
