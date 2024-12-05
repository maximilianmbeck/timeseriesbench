# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Markus Spanring

import subprocess
from pathlib import Path

from omegaconf import DictConfig

from tsbench.ml_utils.run_utils.runner import setup_directory
from tsbench.ml_utils.utils import get_config, get_config_file_from_cli


def run(cfg: DictConfig):
    slurm_dir = setup_directory("slurm", cfg)
    slurm_dir.populate_slurm_template()

    subprocess.Popen(f"sbatch {slurm_dir.slurm_file}", stdout=subprocess.DEVNULL, shell=True)


if __name__ == "__main__":
    cfg_file = get_config_file_from_cli(config_folder="configs", script_file=Path(__file__))
    cfg = get_config(cfg_file)
    run(cfg)
