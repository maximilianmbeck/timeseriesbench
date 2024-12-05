# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import partial

import torch

from ..base import LayerConfigInterface, LayerInterface
from ..init import ParameterInit, ParameterInitConfig
from ..transformer.base import BaseTransformerConfig
from ..utils import create_layer
from .identity import Identity


@dataclass
class PositionalLayerConfig(LayerConfigInterface):
    # should be set via assign_model_config_params
    embedding_dim: int = 0
    max_sequence_length: int = 512

    def assign_model_config_params(self, model_config: BaseTransformerConfig):
        self.embedding_dim = model_config.embedding_dim
        self.max_sequence_length = model_config.context_length


class PositionalEncodingSinusoidalConfig(PositionalLayerConfig):
    pass


@dataclass
class PositionalEmbeddingConfig(PositionalLayerConfig):
    init_embedding: ParameterInitConfig = field(default_factory=lambda: ParameterInitConfig())

    def assign_model_config_params(self, model_config):
        super().assign_model_config_params(model_config)
        assert hasattr(
            model_config, "init_embedding"
        ), f"model_config {model_config.__class__} must have init_embedding attribute"
        self.init_embedding = model_config.init_embedding


class PositionalLayer(LayerInterface):
    @abstractmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        pass


class PositionalEmbedding(PositionalLayer):
    config_class = PositionalEmbeddingConfig

    def __init__(self, config: PositionalEmbeddingConfig):
        super().__init__()
        self.config = config
        self._embedding = torch.nn.Embedding(config.max_sequence_length, config.embedding_dim)
        self.reset_parameters()

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        pos_embeddings = self._embedding.weight[: emb.size(1)]
        return emb + pos_embeddings.unsqueeze(dim=0)

    def reset_parameters(self, **kwargs):
        _init_embedding = ParameterInit(self.config.init_embedding)
        _init_embedding(self._embedding.weight)

    def _create_weight_decay_optim_groups(self) -> tuple[set[torch.nn.Parameter], set[torch.nn.Parameter]]:
        return (), (self._embedding.weight,)


class PositionalEncodingSinusoidal(PositionalLayer):
    config_class = PositionalEncodingSinusoidalConfig

    def __init__(self, config: PositionalEncodingSinusoidalConfig):
        super().__init__()
        self.config = config
        # copied from pytorch.org transformer tutorial
        position = torch.arange(config.max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.embedding_dim, 2) * (-math.log(10000.0) / config.embedding_dim))
        pe = torch.zeros(1, config.max_sequence_length, config.embedding_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        pos_embeddings = self.pe[:, : emb.size(1)]
        return emb + pos_embeddings

    def reset_parameters(self, **kwargs):
        pass


_positional_registry = {
    "positional_encoding_sinusoidal": PositionalEncodingSinusoidal,
    "positional_embedding": PositionalEmbedding,
    "none": Identity,
}


create_positional_layer = partial(create_layer, registry=_positional_registry, layer_cfg_key="positional_layer")
