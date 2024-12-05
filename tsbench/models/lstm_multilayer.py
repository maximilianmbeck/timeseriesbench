# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn

from ..ml_utils.config import NameAndKwargs
from .transformer.lm_transformer import LMTransformer, LMTransformerConfig
from .transformer.sequence_transformer import (
    SequenceTransformer,
    SequenceTransformerConfig,
)


@dataclass
class LMLSTMMultiLayerConfig(LMTransformerConfig):
    positional_layer: NameAndKwargs = field(default_factory=lambda: NameAndKwargs("none"))


class LMLSTMMultiLayer(LMTransformer):
    config_class = LMLSTMMultiLayerConfig

    def __init__(self, config: LMLSTMMultiLayerConfig):
        super().__init__(config)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.embedding_dim,
            num_layers=config.num_layers,
            bias=config.bias,
            batch_first=True,
            dropout=config.dropout,
        )

    def _create_blocks(self, config: LMLSTMMultiLayerConfig) -> None:
        # the multilayer LSTM does not have block structure
        return None

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.lstm.reset_parameters()

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        weight_decay, no_weight_decay = super()._create_weight_decay_optim_groups(**kwargs)
        for pn, p in self.lstm.named_parameters():
            if "bias" in pn:
                no_weight_decay += (p,)
            else:
                weight_decay += (p,)
        return weight_decay, no_weight_decay

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        device = idx.device
        b, t = idx.size()

        assert (
            t <= self.config.context_length
        ), f"Forward input sequence length {t} is longer than context length {self.config.context_length}"

        token_embeddings = self.token_embedding(idx)
        token_embeddings = self.post_embedding_ln(token_embeddings)

        embeddings = self.positional_layer(token_embeddings)
        x = self.emb_dropout(embeddings)

        x, _ = self.lstm(x)

        x = self.blocks_ln(x)
        logits = self.lm_head(x)
        return logits


@dataclass
class SequenceLSTMMultiLayerConfig(SequenceTransformerConfig):
    pass


class SequenceLSTMMultiLayer(SequenceTransformer):
    config_class = SequenceLSTMMultiLayerConfig

    def __init__(self, config: SequenceLSTMMultiLayerConfig):
        super().__init__(config)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.embedding_dim,
            num_layers=config.num_layers,
            bias=config.bias,
            batch_first=True,
            dropout=config.dropout,
        )

    def _create_blocks(self, config: LMLSTMMultiLayerConfig) -> None:
        # the multilayer LSTM does not have block structure
        return None

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.lstm.reset_parameters()

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        weight_decay, no_weight_decay = super()._create_weight_decay_optim_groups(**kwargs)
        for pn, p in self.lstm.named_parameters():
            if "bias" in pn:
                no_weight_decay += (p,)
            else:
                weight_decay += (p,)
        return weight_decay, no_weight_decay

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None, **kwargs) -> torch.Tensor:
        assert (
            x.size(1) <= self.config.context_length
        ), f"Forward input sequence length {x.size(1)} is longer than context length {self.config.context_length}"

        x = self.encoder(x, lengths=lengths)

        x, _ = self.lstm(x)
        x = self.blocks_ln(x)

        x = self.decoder(x, lengths=lengths)

        return x
