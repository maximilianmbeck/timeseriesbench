# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import torch


def laplace_(tensor: torch.Tensor, loc: float, scale: float) -> torch.Tensor:
    with torch.no_grad():
        tensor.copy_(torch.distributions.laplace.Laplace(loc, scale).sample(tensor.shape))
    return tensor
