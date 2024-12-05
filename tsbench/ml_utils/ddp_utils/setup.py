# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import logging

import torch.distributed as dist


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size():
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_rank0():
    return get_rank() == 0


def get_backend():
    if is_distributed():
        return dist.get_backend()
    return None


def log_distributed_config():
    logging.info("------------------")
    logging.info("DIST CONFIG")
    logging.info(f"rank: {get_rank()}")
    logging.info(f"world_size: {get_world_size()}")
    logging.info(f"backend: {get_backend()}")
