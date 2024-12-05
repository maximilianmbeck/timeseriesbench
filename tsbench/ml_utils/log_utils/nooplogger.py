# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import logging
from typing import Any, Dict, List, Union

import torch

from .baselogger import BaseLogger

LOGGER = logging.getLogger(__name__)


class NoOpLogger(BaseLogger):
    """A logger that does nothing. Used as drop-in for distributed training."""

    def __init__(self, **kwargs):
        pass

    def setup_logger(self) -> None:
        pass

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
        specifier: str = "epoch",
    ) -> None:
        pass

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], idx: int, specifier: str = "", name: str = "checkpoint"
    ) -> None:
        pass

    def save_best_checkpoint(self, checkpoint: Dict[str, Any], name: str = "best_checkpoint") -> None:
        pass

    def save_best_checkpoint_idx(self, specifier: str, best_idx: int) -> None:
        pass

    def finish(self, final_results: Dict[str, Any] = {}, exit_code: int = 0):
        pass
