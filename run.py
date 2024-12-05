# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck


import logging
import os
from pathlib import Path

#! IMPORTANT do not import torch here
from tsbench.ml_utils.utils import get_config, get_config_file_and_noderank_from_cli
from tsbench.run_job import run_job

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL)


if __name__ == "__main__":
    import wandb

    print("Starting run ...")
    wandb.login(host="https://wandb.ml.jku.at")
    # wandb.login(host="https://api.wandb.ai")
    cfg_file, node_rank = get_config_file_and_noderank_from_cli(config_folder="configs", script_file=Path(__file__))
    cfg = get_config(cfg_file)
    logging.debug("Running single experiment")
    run_job(cfg=cfg, node_rank=node_rank)
