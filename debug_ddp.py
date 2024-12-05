import os

from torch.distributed import barrier, destroy_process_group, init_process_group
from torch.multiprocessing import spawn


def main():
    devices = "1,2"
    world_size = len(devices.split(","))
    spawn(main_single, nprocs=world_size, args=(devices,))


def main_single(rank, devices):
    device = devices.split(",")[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    import torch

    print(
        f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}' "
        f"device_count={torch.cuda.device_count()} "
        f"rank={rank}"
    )


if __name__ == "__main__":
    main()
