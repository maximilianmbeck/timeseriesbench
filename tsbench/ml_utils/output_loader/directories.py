# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

from dacite import from_dict
from omegaconf import DictConfig, OmegaConf, open_dict

from ..config import SlurmConfig

FN_CONFIG = "config.yaml"
DIR_OUTPUT_FOLDER_NAME = "outputs"


@dataclass
class Directory:
    dir: Union[str, Path] = "."

    config_file_name: str = FN_CONFIG
    log_file_name: str = "output.log"
    _directories: List[Path] = field(default_factory=list)

    def _assign_dir(self) -> None:
        if isinstance(self.dir, JobDirectory):
            self.dir = Path(self.dir.dir).resolve()
        else:
            self.dir = Path(self.dir).resolve()

    def create_directories(self, exist_ok: bool = True) -> None:
        for d in self._directories:
            d.mkdir(exist_ok=exist_ok)

    def is_directory(self) -> bool:
        for d in self._directories:
            if not d.exists():
                raise ValueError(f"Directory {d} does not exist for {self.__class__.__name__}!")
        if not (self.dir / self.config_file_name).exists():
            raise ValueError(f"Config file {self.config_file_name} does not exist for {self.__class__.__name__}!")
        return True

    def save_config(self, config: Union[dict, DictConfig]) -> None:
        OmegaConf.save(config, self.dir / self.config_file_name)

    def load_config(self) -> DictConfig:
        return OmegaConf.load(self.dir / self.config_file_name)

    @property
    def log_file(self) -> Path:
        return self.dir / self.log_file_name

    def __str__(self) -> str:
        return str(self.dir)


@dataclass
class SweepDirectory(Directory):
    jobs_folder_name = DIR_OUTPUT_FOLDER_NAME

    def __post_init__(self):
        self._assign_dir()
        self.jobs_folder = self.dir / self.jobs_folder_name
        self._directories = [self.jobs_folder]

    def __repr__(self) -> str:
        return f"SweepDirectory({str(self)})"


@dataclass
class JobDirectory(Directory):
    stats_folder_name: str = "statistics"
    figures_folder_name: str = "figures"
    checkpoints_folder_name: str = "checkpoints"

    def __post_init__(self):
        self._assign_dir()
        d = self.dir
        self.stats_folder = d / self.stats_folder_name
        self.figures_folder = d / self.figures_folder_name
        self.checkpoints_folder = d / self.checkpoints_folder_name

        self._directories = [self.stats_folder, self.figures_folder, self.checkpoints_folder]

    def __repr__(self) -> str:
        return f"JobDirectory({str(self)})"

    def save_best_checkpoint(
        self, checkpoint: Dict[str, Any], name: str = "best_model", file_extension: str = ".p"
    ) -> None:
        self.save_checkpoint(checkpoint=checkpoint, name=name, file_extension=file_extension)

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        idx: int = -1,
        specifier: str = "",
        name: str = "checkpoint",
        file_extension: str = ".p",
        override_path: Union[str, Path] = "",
    ) -> None:
        """Store checkpoint with pattern `{name}-{specifier}--{idx}{file_extension}`.

        Args:
            specifier (str, optional): Typically the progress measure: step or epoch. Defaults to ''.
            name (str, optional): The Filename. Defaults to 'checkpoint'.
            idx (int, optional): The index. Defaults to -1.
            file_extension (str, optional): Defaults to '.p'.
            override_path (Union[str, Path], optional): Save at this path if specified. Use standard checkpoints folder otherwise.
                                                        Defaults to ''.
        """
        import torch

        filename = self.get_checkpoint_filename(idx=idx, specifier=specifier, name=name, file_extension=file_extension)
        save_path = self.checkpoints_folder
        if override_path:
            save_path = Path(override_path)

        file = save_path / filename
        torch.save(checkpoint, file)

    def get_checkpoint_file(
        self, idx: int = -1, specifier: str = "", name: str = "checkpoint", file_extension: str = ".p"
    ) -> Path:
        # default: load from checkpoints folder
        load_path = self.checkpoints_folder
        # construct filename
        if idx >= 0 and specifier:
            filename = self.get_checkpoint_filename(
                idx=idx, specifier=specifier, name=name, file_extension=file_extension
            )
        elif idx >= 0:
            pattern = f"{name}*--{idx}{file_extension}"
            filenames = [f.name for f in load_path.glob(pattern)]
            assert (
                len(filenames) > 0
            ), f"No checkpoints found for pattern `{pattern}` in directory `{str(load_path)}: {str(filenames)}`!"
            filename = filenames[0]
        else:
            raise ValueError("No valid checkpoint index specified!")

        file = load_path / filename
        return file

    def load_checkpoint(
        self,
        idx: int = -1,
        specifier: str = "",
        name: str = "checkpoint",
        file_extension: str = ".p",
        device: Union[str, int] = "cpu",  # todo: could also by torch.device, but do not want to import torch here
        load_kwargs: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        file = self.get_checkpoint_file(idx=idx, specifier=specifier, name=name, file_extension=file_extension)
        import torch

        from ..torch_utils import get_device

        if not device is None:
            device = get_device(device)

        return torch.load(file, map_location=device, **load_kwargs)

    def get_checkpoint_filename(
        self, idx: int = -1, specifier: str = "", name: str = "checkpoint", file_extension: str = ".p"
    ) -> str:
        """Return the filename in the pattern `{name}-{specifier}--{idx}{file_extension}`."""
        filename = f"{name}"
        if specifier:
            filename += f"-{specifier}"
        if idx >= 0:
            filename += f"--{idx}"
        filename += file_extension
        return filename

    def get_checkpoint_indices(self) -> List[int]:
        idxes = [int(f.stem.split("--")[-1]) for f in self.checkpoints_folder.iterdir() if len(f.stem.split("--")) > 1]
        idxes.sort()
        return idxes

    @staticmethod
    def load_resume_checkpoint(job_dir: str, checkpoint_idx: int, device: Union[str, int] = "cpu") -> Dict[str, Any]:
        job_directory = JobDirectory(job_dir)
        checkpoint = job_directory.load_checkpoint(idx=checkpoint_idx, device=device)
        return checkpoint


@dataclass
class SlurmDirectory(JobDirectory):
    slurm_template_dir: str = "config/slurm"
    slurm_file_name: str = "slurm_submit.sh"
    config: DictConfig = None
    slurm_config: SlurmConfig = None

    def _get_slurm_template(self):
        template_path = Path(self.slurm_template_dir) / "karolina.sh"
        if "karolina" in os.environ.get("HOSTNAME", ""):
            template_path = Path(self.slurm_template_dir) / "karolina.sh"
        if "meluxina" in os.environ.get("HOSTNAME", ""):
            template_path = Path(self.slurm_template_dir) / "meluxina.sh"
        if "leonardo" in os.environ.get("HOSTNAME", ""):
            template_path = Path(self.slurm_template_dir) / "leonardo.sh"

        assert template_path.exists(), f"No template found for cluster {os.environ['HOSTNAME']}"

        with open(template_path, "r") as f:
            return f.read()

    def populate_slurm_template(self):
        self.slurm_config = from_dict(data_class=SlurmConfig, data=OmegaConf.to_container(self.config.slurm))

        settings = {
            "config_path": self.dir,
            "config_file": self.config_file_name,
            "account": self.slurm_config.account,
            "nodes": self.slurm_config.nodes,
            "time": self.slurm_config.time,
            "partition": self.slurm_config.partition,
            "env_name": self.slurm_config.env_name,
            "chdir": Path().cwd(),
            "output": self.dir,
        }

        template = self._get_slurm_template()
        slurm_config = template.format(**settings)

        with open(self.slurm_file, "w") as f:
            f.write(slurm_config)

    def save_config(self, config: Union[dict, DictConfig]) -> None:
        config.config.ddp = {}
        config.config.ddp.n_nodes = config.slurm.nodes
        config.config.ddp.devices = "${oc.env:CUDA_VISIBLE_DEVICES}"
        config.config.ddp.master_addr = "${oc.env:MASTER_ADDR}"
        config.config.ddp.master_port = "${oc.env:MASTER_PORT}"

        OmegaConf.save(config, self.dir / self.config_file_name)

    @property
    def slurm_file(self) -> Path:
        return self.dir / self.slurm_file_name
