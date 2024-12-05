# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import torch

from ..base import LayerInterface


class Identity(torch.nn.Identity, LayerInterface):
    def reset_parameters(self, **kwargs):
        pass

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)
