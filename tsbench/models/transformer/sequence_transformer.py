# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from ...ml_utils.config import NameAndKwargs
from ..base import LayerInterface
from .base import BaseTransformer, BaseTransformerConfig


@dataclass
class SequenceTransformerConfig(BaseTransformerConfig):
    encoder: NameAndKwargs = None
    decoder: NameAndKwargs = None


class SequenceTransformer(BaseTransformer):
    config_class = SequenceTransformerConfig

    def __init__(self, config: SequenceTransformerConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config

        self.encoder = self._create_encoder(config)
        self.decoder = self._create_decoder(config)

    @property
    def input_dim(self) -> Sequence[int]:
        return self.config.input_dim

    @property
    def output_dim(self) -> Sequence[int]:
        return self.config.output_dim

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        weight_decay, no_weight_decay = super()._create_weight_decay_optim_groups(**kwargs)
        wd, no_wd = self._get_weight_decay_optim_groups_for_modules(modules=[self.encoder, self.decoder], **kwargs)
        weight_decay += wd
        no_weight_decay += no_wd
        return weight_decay, no_weight_decay

    def _create_encoder(self, config: SequenceTransformerConfig) -> LayerInterface:
        from ..seq_enc_dec import create_encoder

        return create_encoder(config=config)

    def _create_decoder(self, config: SequenceTransformerConfig) -> LayerInterface:
        from ..seq_enc_dec import create_decoder

        return create_decoder(config=config)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None, **kwargs) -> torch.Tensor:
        assert (
            x.size(1) <= self.config.context_length
        ), f"Forward input sequence length {x.size(1)} is longer than context length {self.config.context_length}"

        x = self.encoder(x, lengths=lengths)

        for block in self.blocks:
            x = block(x, lengths=lengths)
        x = self.blocks_ln(x)

        x = self.decoder(x, lengths=lengths)

        return x
