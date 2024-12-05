# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from dataclasses import dataclass

import torch
from torch import nn

from ..base import LayerInterface
from ..components.act_fn import get_act_fn
from ..components.linear import Linear, LinearConfig
from .base import BaseFeedForwardConfig


@dataclass
class FeedForwardConfig(BaseFeedForwardConfig):
    proj_factor: int = 4


class FeedForward(LayerInterface):
    """Simple Feed Forward NN used in standard Transformer."""

    config_class = FeedForwardConfig

    def __init__(
        self,
        config: FeedForwardConfig,
    ):
        super().__init__()
        self.config = config
        embedding_dim = config.embedding_dim
        dropout = config.dropout
        proj_factor = config.proj_factor
        bias = config.bias
        self.ff_proj_up = Linear(
            LinearConfig(
                in_features=embedding_dim, out_features=proj_factor * embedding_dim, bias=bias, init=config.proj_up_init
            )
        )
        self.ff_proj_down = Linear(
            LinearConfig(
                in_features=proj_factor * embedding_dim,
                out_features=embedding_dim,
                bias=bias,
                init=config.proj_down_init,
            )
        )
        self.dropout = nn.Dropout(dropout)
        self.act_fn = get_act_fn(self.config.act_fn)

    def reset_parameters(self, block_idx: int = None, num_blocks: int = None):
        self.config.set_std_init_scaling_downproj(num_blocks=num_blocks)
        self.ff_proj_up.reset_parameters()
        self.ff_proj_down.reset_parameters()

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[set[nn.Parameter], set[nn.Parameter]]:
        return self._get_weight_decay_optim_groups_for_modules(modules=[self.ff_proj_up, self.ff_proj_down], **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.ff_proj_down(self.act_fn(self.ff_proj_up(x))))
        # x = self.ff_act_fn(x)
        # x = self.ff_proj_down(x)
        # x = self.ff_dropout(x)
        return x
