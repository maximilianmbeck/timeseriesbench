# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
from torch import nn

from ...ml_utils.config import NameAndKwargs
from ..base import BaseSequenceModelConfig, BaseSequenceModelTrain
from ..components.ln import NormConfig, create_norm_layer

LOGGER = logging.getLogger(__name__)


@dataclass
class BaseTransformerConfig(BaseSequenceModelConfig):
    num_layers: int = 1
    embedding_dim: int = 128
    bias: bool = True
    dropout: float = 0.0
    block: NameAndKwargs = None
    norm: NormConfig = field(default_factory=lambda: NormConfig(norm_type="layer"))

    def __post_init__(self):
        self.norm.bias = self.bias


class BaseTransformer(BaseSequenceModelTrain):
    def __init__(self, config: BaseTransformerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.blocks = self._create_blocks(config)
        self.blocks_ln = create_norm_layer(config=config.norm, ndim=config.embedding_dim)

    @property
    def context_length(self) -> int:
        return self.config.context_length

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    def reset_parameters(self) -> None:
        if isinstance(self.blocks, nn.ModuleList):
            for block in self.blocks:
                block.reset_parameters()
        self.blocks_ln.reset_parameters()

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        modules = [self.blocks_ln]
        if self.blocks is not None:
            modules += self.blocks
        return self._get_weight_decay_optim_groups_for_modules(modules=modules, **kwargs)

    def _create_blocks(self, config: BaseTransformerConfig) -> Optional[nn.ModuleList]:
        from .blocks import create_block

        return nn.ModuleList([create_block(config=config, block_idx=i) for i in range(config.num_layers)])

    # def get_loss_func(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    #     import torch.nn.functional as F

    #     def loss_fn(logits, targets):
    #         assert not torch.any(torch.isnan(logits.view(-1)))
    #         assert not torch.any(torch.isnan(targets.view(-1)))
    #         loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    #         return loss

    #     return loss_fn
