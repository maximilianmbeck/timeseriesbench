# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import torch
import torch.distributed as dist

from .setup import get_world_size, is_distributed


def all_gather_tensor_list(tensor_list: list[torch.Tensor]) -> list[torch.Tensor]:
    """(All) gather a list of tensors from all processes.

    Args:
        tensor_list (list[torch.Tensor]): List of tensors to all gather.

    Returns:
        list[torch.Tensor]: All gathered list of tensors.
    """
    if not is_distributed():
        return tensor_list

    stacked_tensor = torch.stack(tensor_list)

    all_rank_tensors = [torch.zeros_like(stacked_tensor) for _ in range(get_world_size())]
    dist.all_gather(tensor=stacked_tensor, tensor_list=all_rank_tensors)
    all_rank_tensor = torch.cat(all_rank_tensors, dim=0)
    all_gathered_tensor_list = list(all_rank_tensor)

    return all_gathered_tensor_list
