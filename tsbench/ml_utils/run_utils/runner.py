# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type, Union

import dacite
from omegaconf import DictConfig, OmegaConf, open_dict

from ..log_utils.log_cmd import setup_logging
from ..output_loader.directories import (
    Directory,
    JobDirectory,
    SlurmDirectory,
    SweepDirectory,
)
from ..utils import setup_experiment_dir

LOGGER = logging.getLogger(__name__)

DIR_BASE = "."  # use the working directory of the run script as base dir


class Runner(ABC):
    def __init__(self, runner_dir: Union[str, Path] = ""):
        self.runner_dir = Path(runner_dir)

    @abstractmethod
    def run(self) -> None:
        pass


def setup_directory(run_type: str, config: DictConfig, configure_logging: bool = True) -> Directory:
    exp_dir = config.config.experiment_data.get("experiment_dir", None)

    if "slurm" in config and exp_dir is not None:
        exp_dir = Path(exp_dir)
    else:
        exp_dir = setup_experiment_dir(
            experiment_name=config.config.experiment_data.experiment_name,
            base_dir=config.config.experiment_data.get("output_dir", None),
        )

    with open_dict(config):
        config.config.experiment_data.experiment_dir = str(exp_dir.resolve())
        config.config.experiment_data.job_name = exp_dir.name

    if run_type == "sweep":
        exp_dir = SweepDirectory(dir=exp_dir)
    elif run_type == "job":
        exp_dir = JobDirectory(dir=exp_dir)
    elif run_type == "slurm":
        exp_dir = SlurmDirectory(dir=exp_dir, config=config)

    exp_dir.create_directories()
    exp_dir.save_config(config)

    if configure_logging:
        setup_logging(logfile=exp_dir.log_file)
        LOGGER.info(f"Starting {run_type} with config: \n{OmegaConf.to_yaml(config)}")

    return exp_dir


def run_job(cfg: DictConfig, trainer_class: Type[Runner]) -> Path:
    OmegaConf.resolve(cfg)
    setup_directory("job", cfg)
    if cfg.get("convert_to_dataclass", True):
        from dacite import from_dict

        config = from_dict(
            data_class=trainer_class.config_class,
            data=OmegaConf.to_container(cfg.config),
            config=dacite.Config(strict=True),
        )
    else:
        config = cfg.config
    trainer = trainer_class(config=config)
    trainer.run()
    return trainer.runner_dir


def run_sweep(cfg: DictConfig) -> None:
    sweep_dir = setup_directory("sweep", cfg)
    LOGGER.info(f"Starting experiment with config: \n{OmegaConf.to_yaml(cfg)}")
    # absolute path to run script
    script_path = Path().cwd() / "run.py"
    # all runs started with this sweep should go to the jobs folder
    os.chdir(sweep_dir.dir)
    from ...ml_utils.run_utils.run_handler import RunHandler

    run_handler = RunHandler(sweep_dir=sweep_dir.dir.resolve(), config=cfg, script_path=script_path)
    run_handler.run()
    return run_handler.runner_dir
