# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import os
import sys
from typing import Any, Sequence

import torch.distributed as dist

MASTER_ADDR = "localhost"
MASTER_PORT = "12334"
BACKEND = "nccl"
INIT_METHOD = "env://"

DEVICES = "2,3,4,5"


def _world_size(devices: str = DEVICES) -> int:
    return len(devices.split(","))


def _device_for_rank(devices: str = DEVICES) -> Sequence[str]:
    return devices.split(",")


def _is_distributed():
    return dist.is_available() and dist.is_initialized()


def _get_rank():
    if _is_distributed():
        return dist.get_rank()
    return 0


def print_rank_zero(msg: str) -> None:
    if _get_rank() == 0:
        print(f"[R0] {msg}")


def print_rank(msg: str) -> None:
    print(f"[R{_get_rank()}] {msg}")


def run_distributed_job(rank_args: Sequence[dict[str, Any]] = {}) -> None:
    from torch.multiprocessing import spawn

    spawn(run_multiprocess, args=(rank_args,), nprocs=_world_size())


def run_multiprocess(rank: int, rank_args: Sequence[dict[str, Any]]) -> None:
    from torch.distributed import barrier, destroy_process_group, init_process_group

    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    os.environ["CUDA_VISIBLE_DEVICES"] = _device_for_rank()[rank]
    import torch

    print(
        f"[R{rank}] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} device_count={torch.cuda.device_count()}"
    )

    init_process_group(backend=BACKEND, init_method=INIT_METHOD, world_size=_world_size(), rank=rank)

    print_rank_zero("### RUNNING SINGLE RANKS:")
    barrier()
    run_single_rank(**rank_args[rank])
    destroy_process_group()


def run_single_rank(val, val_list, **kwargs: Any):
    """Run a single rank."""
    import torch

    print_rank(f"val: {val}")
    val = val.cuda()

    #! all reduce
    # reduced_val = torch.zeros_like(val)
    reduced_val = val.clone()
    dist.all_reduce(reduced_val, op=dist.ReduceOp.SUM)
    print_rank(f"reduced val: {reduced_val}")

    #! gather
    dest_rank = 0
    gathered_vals = None
    # WRONG:
    # dist.gather(tensor=val, gather_list=gathered_vals, dst=dest_rank)
    # -> ValueError: Argument ``gather_list`` must NOT be specified on non-destination ranks.
    # CORRECT:
    if _get_rank() == dest_rank:
        gathered_vals = [torch.zeros_like(val) for _ in range(_world_size())]
        dist.gather(tensor=val, gather_list=gathered_vals, dst=dest_rank)
    else:
        dist.gather(tensor=val, dst=dest_rank)
    print_rank(f"gathered vals: {gathered_vals}")

    #! all gather
    all_gathered_vals = [torch.zeros_like(val) for _ in range(_world_size())]
    dist.all_gather(tensor=val, tensor_list=all_gathered_vals)
    print_rank(f"all gathered vals: {all_gathered_vals}")

    #! all gather object
    all_gathered_vals = [None for _ in range(_world_size())]
    dist.all_gather_object(obj=val.item(), object_list=all_gathered_vals)
    print_rank(f"all gathered object vals: {all_gathered_vals}")

    #! own: all gather tensor list
    print_rank(f"val_list: {val_list}")
    val_list = [v.cuda() for v in val_list]

    sys.path.append("PATH_TO_THIS_REPO")  # TODO
    from ...ml_utils.ddp_utils.primitives import all_gather_tensor_list

    all_gathered_list = all_gather_tensor_list(val_list)
    print_rank(f"all gathered list: {all_gathered_list}")


def setup_rank_args() -> Sequence[dict[str, Any]]:
    """Setup rank args for each rank."""
    import torch

    rank_args = []
    for i in range(_world_size()):
        rank_args.append(
            {
                "val": torch.tensor(i + 1.0, dtype=torch.bfloat16),
                "val_list": [
                    torch.tensor(i + 1.0, dtype=torch.bfloat16),
                    torch.tensor(i + 1.0 + _world_size(), dtype=torch.bfloat16),
                ],
            }
        )

    return rank_args


if __name__ == "__main__":
    rank_args = setup_rank_args()

    run_distributed_job(rank_args=rank_args)
