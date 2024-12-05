# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import logging
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn

from ...ml_utils.config import NameAndKwargs
from ..components.identity import Identity
from ..components.linear import Linear, LinearConfig
from ..components.ln import create_norm_layer
from ..components.positional import create_positional_layer
from ..init import ParameterInit, ParameterInitConfig
from .base import BaseTransformer, BaseTransformerConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class LMTransformerConfig(BaseTransformerConfig):
    tie_weights: bool = True
    add_post_embedding_ln: bool = False
    add_embedding_dropout: bool = True
    init_embedding: ParameterInitConfig = field(
        default_factory=lambda: ParameterInitConfig(method="normal", range=0.02)
    )

    positional_layer: NameAndKwargs = field(default_factory=lambda: NameAndKwargs("positional_embedding"))


class LMTransformer(BaseTransformer):
    config_class = LMTransformerConfig

    def __init__(self, config: LMTransformerConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self.config = config

        self.token_embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim)
        self.positional_layer = create_positional_layer(config)
        self.emb_dropout = nn.Dropout(config.dropout) if config.add_embedding_dropout else nn.Identity()

        if config.add_post_embedding_ln:
            self.post_embedding_ln = create_norm_layer(ndim=config.embedding_dim, config=config.norm)
        else:
            self.post_embedding_ln = Identity()

        self.lm_head = Linear(
            LinearConfig(in_features=config.embedding_dim, out_features=config.vocab_size, bias=False)
        )
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def reset_parameters(self):
        super().reset_parameters()

        _init_embedding = ParameterInit(config=self.config.init_embedding)
        _init_embedding(self.token_embedding.weight)
        self.positional_layer.reset_parameters()
        self.post_embedding_ln.reset_parameters()

        if not self.config.tie_weights:
            self.lm_head.reset_parameters()

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        weight_decay, no_weight_decay = super()._create_weight_decay_optim_groups(**kwargs)
        no_weight_decay += (self.token_embedding.weight,)

        modules = [self.positional_layer, self.post_embedding_ln]
        if not self.config.tie_weights:
            modules.append(self.lm_head)
        wd, no_wd = self._get_weight_decay_optim_groups_for_modules(modules=modules, **kwargs)
        weight_decay += wd
        no_weight_decay += no_wd
        return weight_decay, no_weight_decay

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        device = idx.device
        b, t = idx.size()

        assert (
            t <= self.config.context_length
        ), f"Forward input sequence length {t} is longer than context length {self.config.context_length}"

        token_embeddings = self.token_embedding(idx)
        # ? Note LayerNorm is applied before positional embedding!
        token_embeddings = self.post_embedding_ln(token_embeddings)

        embeddings = self.positional_layer(token_embeddings)
        x = self.emb_dropout(embeddings)

        for i, block in enumerate(self.blocks):
            x = block(x)

        x = self.blocks_ln(x)
        logits = self.lm_head(x)
        return logits
