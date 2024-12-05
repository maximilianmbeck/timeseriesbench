# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck


import logging
import os
import sys

from dacite import from_dict
from omegaconf import DictConfig, OmegaConf

os.environ["CUDA_PATH"] = sys.prefix
os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(":4096:8")  # make cublas use deterministic algorithms

#! IMPORTANT: we should not import torch in any other place than here, otherwise it will break DDP
# explanation: If we import torch before spawning multiple processes, torch will initialize cuda with
# all visible devices, which will cause problems when spawning multiple processes with DDP, since
# each process sets its own CUDA_VISIBLE_DEVICES environment variable.
from .config import Config
from .ml_utils.ddp_utils.setup import get_rank
from .ml_utils.log_utils.log_cmd import setup_logging_multiprocess
from .ml_utils.output_loader.directories import Directory
from .ml_utils.run_utils.runner import setup_directory

LOGGER = logging.getLogger(__name__)


OmegaConf.register_new_resolver("eval", eval)


def run_job(cfg: DictConfig, node_rank: int = 0) -> None:
    """Main entry point for running a single job."""
    OmegaConf.resolve(cfg)
    # NOTE: we already configure logging before spawning multiple processes in order to log
    # exception to the log file
    job_dir = setup_directory("job", cfg, configure_logging=True)
    cfg: Config = from_dict(data_class=Config, data=OmegaConf.to_container(cfg.config))  # TODO set strict to True

    if cfg.ddp is not None and cfg.ddp.enable_ddp:
        from torch.multiprocessing import spawn

        LOGGER.info(
            f"Running DDP with world size {cfg.ddp.world_size}: {cfg.ddp.n_nodes} node(s) and {cfg.ddp.n_procs_per_node} processes per node"
        )
        LOGGER.info(f"{cfg.ddp}")
        spawn(run_multiprocess, args=(cfg, job_dir, node_rank), nprocs=cfg.ddp.ranks_per_node)
    else:
        run_single(cfg, job_dir)


def run_multiprocess(local_rank: int, cfg: Config, job_dir: Directory, node_rank: int = 0) -> None:
    from torch.distributed import barrier, destroy_process_group, init_process_group

    assert node_rank >= 0
    assert node_rank < cfg.ddp.n_nodes, f"node_rank={node_rank} >= n_nodes={cfg.ddp.n_nodes}"

    cfg.ddp.node_rank = node_rank
    cfg.ddp.local_rank = local_rank
    # determine global rank
    global_rank = cfg.ddp.node_rank * cfg.ddp.n_procs_per_node + cfg.ddp.local_rank
    cfg.ddp.global_rank = global_rank

    # setup logging multiprocess
    os.environ["MASTER_ADDR"] = cfg.ddp.master_addr
    os.environ["MASTER_PORT"] = str(cfg.ddp.master_port)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.ddp.gpu_id_for_global_rank(global_rank))
    import torch

    LOGGER.info(
        f"global_rank={global_rank} "
        f"node_rank={node_rank} "
        f"local_rank={local_rank} "
        f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
        f"device_count={torch.cuda.device_count()} "
    )

    init_process_group(
        backend=cfg.ddp.backend, init_method=cfg.ddp.init_method, world_size=cfg.ddp.world_size, rank=global_rank
    )

    assert cfg.ddp.global_rank == get_rank(), f"global_rank={cfg.ddp.global_rank} != rank={get_rank()}"
    LOGGER.info(
        f"[Rank: {cfg.ddp.global_rank}] node_rank={cfg.ddp.node_rank}, local_rank={cfg.ddp.local_rank}, gpu_id: {cfg.ddp.gpu_id_for_global_rank(global_rank)}"
    )

    setup_logging_multiprocess(logfile=job_dir.log_file)
    barrier()
    from .main import main

    main(cfg)
    destroy_process_group()


def run_single(cfg: Config, job_dir: Directory) -> None:
    from .main import main

    main(cfg)
