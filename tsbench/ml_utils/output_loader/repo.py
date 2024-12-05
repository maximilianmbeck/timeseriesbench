# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Union

from omegaconf import DictConfig, OmegaConf

from ..run_utils.run_handler import SWEEP_KEY
from ..time_utils import FORMAT_DATETIME_MID
from .result_loader import JobResult, SweepResult

KEY_CFG_CREATED = "__config_created"
KEY_CFG_UPDATED = "__config_last_updated"
FORMAT_CFG_DATETIME = FORMAT_DATETIME_MID

RUN_FN = "run.py"
RUN_SWEEP_FN = "run_sweep.py"

SLURM_KEY = "slurm"
RUN_SLURM_FN = "run_slurm.py"


@dataclass
class Repo:
    """A class abstracting a repository folder.
    Provides access to experiment configs and experiment outputs."""

    dir: Path
    hydra_defaults: DictConfig = field(default_factory=lambda: OmegaConf.create({}))

    def __post_init__(self):
        self.dir = self.dir.resolve()

    @property
    def config_dir(self):
        return self.dir / "configs"

    @property
    def output_dir(self):
        return self.dir / "outputs"

    def create_experiment(self, cfg: DictConfig, override: bool = False) -> str:
        """Convenience function. Wrapper for `create_config()`.
        Instead of the Path to the config file it returns the run command.

        Args:
            cfg (DictConfig): Experiment config.
            override (bool, optional): Override existing experiment config. Defaults to False.

        Raises:
            FileExistsError: The experiment config already exists and override is False.

        Returns:
            str: The run command.
        """
        if hasattr(cfg, "run_config") and hasattr(cfg.run_config, "gpu_ids"):
            gpu_ids = cfg.run_config.gpu_ids
            prefix = "CUDA_VISIBLE_DEVICES=" + ",".join(map(str, gpu_ids))
        else:
            prefix = ""
        config_file = self.create_config(cfg, override)
        # sweep or normal run
        run_file = RUN_FN
        if SWEEP_KEY in cfg:
            run_file = RUN_SWEEP_FN

        if SLURM_KEY in cfg:
            return f"python {RUN_SLURM_FN} --config-name {config_file.name}"

        return f"{prefix} python {run_file} --config-name {config_file.name}"

    def create_config(self, cfg: DictConfig, override: bool = False) -> Path:
        """Create a config.yaml for a new experiment.

        Args:
            cfg (DictConfig): Experiment config.
            override (bool, optional): Override existing experiment config. Defaults to False.

        Raises:
            FileExistsError: The experiment config already exists and override is False.

        Returns:
            Path: Path to the config file.
        """
        filename = cfg.config.experiment_data.experiment_name + ".yaml"
        config_file = self.config_dir / filename
        now = datetime.now().strftime(FORMAT_CFG_DATETIME)
        if not config_file.exists():
            # add timestamp and last updated to config
            cfg[KEY_CFG_CREATED] = now
            cfg[KEY_CFG_UPDATED] = now
        else:
            if override:
                existing_cfg = OmegaConf.load(config_file)
                old_created = existing_cfg[KEY_CFG_CREATED]
                cfg[KEY_CFG_CREATED] = old_created
                cfg[KEY_CFG_UPDATED] = now
            else:
                raise FileExistsError(
                    f"The config `{filename}` already exists at `{self.config_dir!s}`!\nSet `override=True` to override the existing file."
                )
        # add hydra defaults before saving
        save_cfg = OmegaConf.merge(self.hydra_defaults, cfg)
        OmegaConf.save(save_cfg, config_file)
        return config_file

    def get_output_loader(self, cfg: DictConfig, additional_search_pattern: str = "") -> Union[SweepResult, JobResult]:
        """Create an output loader for a experiment (either sweep or single job).

        Args:
            cfg (DictConfig): The config file for the experiment.
            additional_search_pattern (str, optional): Search pattern to disambiguate between multiple run output folders.
                                                       Defaults to ''.

        Raises:
            FileNotFoundError: Experiment output folder does not exist.
            ValueError: There are multiple experiment folder for the config.

        Returns:
            Union[SweepResult, JobResult]: The output loader.
        """
        glob_pattern = (
            f"{cfg.config.experiment_data.experiment_name}*"
            if not additional_search_pattern
            else f"{cfg.config.experiment_data.experiment_name}*{additional_search_pattern}*"
        )
        output_dirs = list(self.output_dir.glob(pattern=glob_pattern))
        if len(output_dirs) < 1:
            raise FileNotFoundError(f"No output folder is matching pattern `{glob_pattern}`!")
        elif len(output_dirs) > 1:
            raise ValueError(f"Found multiple output folders for pattern `{glob_pattern}`:\n{output_dirs}")

        assert len(output_dirs) == 1
        from . import create_job_output_loader

        return create_job_output_loader(output_dirs[0])
