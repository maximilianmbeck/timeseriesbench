# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import logging
import os
import sys
from typing import Iterable, Union

from tqdm import tqdm

from ..ddp_utils.setup import get_rank, is_rank0

FORMAT_LOGGING_RANK = "[%(asctime)s][%(name)s][%(levelname)s][R{:d}] - %(message)s"

LOGGER = logging.getLogger(__name__)


def get_loglevel() -> str:
    """Get loglevel from environment variable.

    Returns:
        str: loglevel
    """
    return os.environ.get("LOGLEVEL", "INFO").upper()


def get_tqdm_pbar(iterable: Iterable, desc: str = "", **kwargs):
    if is_rank0() and get_loglevel() == "INFO":
        return tqdm(iterable, desc=desc, **kwargs)
    return iterable


def setup_logging(logfile: str = "output.log"):
    """Initialize logging to `log_file` and stdout.

    Args:
        log_file (str, optional): Name of the log file. Defaults to "output.log".
    """
    from ...ml_utils.log_utils.baselogger import FORMAT_LOGGING

    file_handler = logging.FileHandler(filename=logfile)
    stdout_handler = logging.StreamHandler(sys.stdout)

    # retrieve loglevel from external environment variable, default is INFO
    LOGLEVEL = get_loglevel()
    logging.basicConfig(
        handlers=[file_handler, stdout_handler],
        level=LOGLEVEL,
        format=FORMAT_LOGGING,
        force=True,
    )

    setup_exception_logging()

    LOGGER.info(f"Logging to {logfile} initialized.")


def setup_exception_logging():
    """Make sure that uncaught exceptions are logged with the logging."""

    # Log uncaught exceptions
    def exception_logging(typ, value, traceback):
        LOGGER.exception("Uncaught exception", exc_info=(typ, value, traceback))

    sys.excepthook = exception_logging


def setup_logging_multiprocess(logfile: str = "output.log"):
    """Initialize logging to `log_file` and stdout for torch distributed training.

    Args:
        log_file (str, optional): Name of the log file. Defaults to "output.log".
    """

    file_handler = logging.FileHandler(filename=logfile)
    stdout_handler = logging.StreamHandler(sys.stdout)

    def _setup_subprocess_logger(rank: int, level: Union[str, int]) -> None:
        logger = logging.getLogger()
        logger.setLevel(level)
        formatter = logging.Formatter(FORMAT_LOGGING_RANK.format(rank))
        stdout_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

    # retrieve loglevel from external environment variable, default is INFO
    LOGLEVEL = get_loglevel()
    # give all processes the same handlers
    logging.basicConfig(
        handlers=[stdout_handler, file_handler],
        force=True,
    )
    # set log format (adding the rank) and loglevel depending on rank
    if is_rank0():
        # standard logger should log everything
        _setup_subprocess_logger(get_rank(), logging.getLevelName(LOGLEVEL))
    else:
        # other processes should only log CRITICAL and higher
        _setup_subprocess_logger(get_rank(), logging.ERROR)

    # TODO setup exception logging for multiprocess
    # setup_exception_logging()  # exceptions are not logged to file yet, check how to do this with multiprocessing
    # sys except hook does not work with multiprocessing (see [1]) (Remedy see [2])
    # [1] https://stackoverflow.com/questions/47815850/python-sys-excepthook-on-multiprocess
    # [2] https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
    # Note: this issue can be resolved by configuring logging in the main process before spawning and setup exception logging there

    LOGGER.info(f"Logging to {logfile} initialized.")
