# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import torch

from ..output_loader.directories import JobDirectory
from ..utils import convert_dict_to_python_types, save_dict_as_yml
from .baselogger import (
    FN_DATA_LOG,
    FN_FINAL_RESULTS,
    LOG_STEP_KEY,
    BaseLogger,
    increment_log_step_on_call,
)

LOGGER = logging.getLogger(__name__)


class FileLogger(BaseLogger):
    """A logger that logs to disc only."""

    def __init__(self, log_dir: Union[str, Path, JobDirectory], **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def setup_logger(self):
        """Creates directories."""
        self.logger_directory.create_directories()
        LOGGER.info(f"Created log directories in {self.logger_directory.dir}.")

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
        """Logs keys and values to disc."""
        if keys_multiple_vals or keys_val:
            log_dict = self._create_log_dict(epoch, train_step, keys_multiple_vals, keys_val)

            log_dict_numeric_vals = convert_dict_to_python_types(log_dict)
            if log_to_console:
                # log to console
                LOGGER.info(f"{prefix} \n{pd.Series(log_dict_numeric_vals, dtype=float)}")
            else:
                LOGGER.debug(f"{prefix} \n{pd.Series(log_dict_numeric_vals, dtype=float)}")

            # add the internal log_step to every log
            log_dict_numeric_vals.update({LOG_STEP_KEY: self.log_step})
            # add the log dict to the logs
            self.logs[prefix].append(log_dict_numeric_vals)

    def finish(self, final_results: Dict[str, Any] = {}, exit_code: int = 0):
        LOGGER.info(f"Logging finished. Saving logs to {self.logger_directory.dir}.")

        if final_results:
            save_dict_as_yml(self.logger_directory.stats_folder, filename=FN_FINAL_RESULTS, dictionary=final_results)

        # save data to disc
        for datasource, log_data in self.logs.items():
            filename = FN_DATA_LOG.format(datasource=datasource)
            LOGGER.info(f"Creating dump: {filename}")
            log_data_df = pd.DataFrame(log_data)
            save_path = self.logger_directory.stats_folder / filename
            with save_path.open("wb") as f:
                log_data_df.to_csv(f)
