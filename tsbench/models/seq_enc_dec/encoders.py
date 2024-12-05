# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn

from ...ml_utils.config import NameAndKwargs
from ..base import LayerConfigInterface, LayerInterface
from ..components.ln import NormConfig, create_norm_layer
from ..components.positional import Identity, create_positional_layer
from ..init import ParameterInit, ParameterInitConfig
from ..transformer.base import BaseTransformerConfig


@dataclass
class LinearEncoderConfig(LayerConfigInterface):
    # will be assigned from base model config
    input_dim: Sequence[int] = None
    embedding_dim: int = None
    context_length: int = None

    def assign_model_config_params(self, model_config: BaseTransformerConfig):
        self.context_length = model_config.context_length
        self.embedding_dim = model_config.embedding_dim
        self.input_dim = model_config.input_dim


class LinearEncoder(LayerInterface):
    config_class = LinearEncoderConfig

    def __init__(self, config: LinearEncoderConfig):
        super().__init__()
        self.config = config

        self.flatten = nn.Flatten(start_dim=-len(config.input_dim)) if len(config.input_dim) > 1 else nn.Identity()
        flat_input_dim = torch.prod(torch.tensor(config.input_dim)).item()
        self.encoder = nn.Linear(flat_input_dim, config.embedding_dim)

    def reset_parameters(self, **kwargs):
        self.encoder.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # shape: (batch_size, context_length, (input_dim))
        x = self.flatten(x)
        # shape: (batch_size, context_length, flat_input_dim)
        x = self.encoder(x)
        # shape: (batch_size, context_length, embedding_dim)
        return x


# embedding encoder contains:
# - token embedding
# - optional post embedding ln
# - positional embedding
# - embedding dropout


@dataclass
class EmbeddingEncoderConfig(LayerConfigInterface):
    add_post_embedding_ln: bool = False
    positional_layer: NameAndKwargs = field(default_factory=lambda: NameAndKwargs(name="none"))
    init_embedding: ParameterInitConfig = field(
        default_factory=lambda: ParameterInitConfig(method="normal", range=0.02)
    )

    # will be assigned from base model config
    context_length: int = None
    embedding_dim: int = None
    vocab_size: int = None
    dropout: float = 0.0
    bias: bool = True
    norm: NormConfig = field(default_factory=lambda: NormConfig(norm_type="layer"))

    def assign_model_config_params(self, model_config: BaseTransformerConfig):
        self.context_length = model_config.context_length
        self.embedding_dim = model_config.embedding_dim
        self.vocab_size = model_config.vocab_size
        self.dropout = model_config.dropout
        self.bias = model_config.bias
        self.norm = model_config.norm
        self.norm.bias = self.bias

    def __post_init__(self):
        self.norm.bias = self.bias


class EmbeddingEncoder(LayerInterface):
    config_class = EmbeddingEncoderConfig

    def __init__(self, config: EmbeddingEncoderConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim)
        self.positional_layer = create_positional_layer(config)
        self.emb_dropout = nn.Dropout(config.dropout)
        if config.add_post_embedding_ln:
            self.post_embedding_ln = create_norm_layer(ndim=config.embedding_dim, config=config.norm)
        else:
            self.post_embedding_ln = Identity()

    def reset_parameters(self, **kwargs):
        _init_embedding = ParameterInit(self.config.init_embedding)
        _init_embedding(self.token_embedding.weight)
        self.positional_layer.reset_parameters()
        self.post_embedding_ln.reset_parameters()

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[set[nn.Parameter], set[nn.Parameter]]:
        weight_decay, no_weight_decay = self._get_weight_decay_optim_groups_for_modules(
            modules=[self.positional_layer, self.post_embedding_ln], **kwargs
        )
        no_weight_decay += (self.token_embedding.weight,)
        return weight_decay, no_weight_decay

    def forward(self, idx: torch.Tensor, **kwargs) -> torch.Tensor:
        b, t = idx.shape
        assert (
            t <= self.config.context_length
        ), f"Forward input sequence length {t} is longer than context length {self.config.context_length}"

        token_embeddings = self.token_embedding(idx)
        token_embeddings = self.post_embedding_ln(token_embeddings)
        embeddings = self.positional_layer(token_embeddings)

        x = self.emb_dropout(embeddings)
        return x
