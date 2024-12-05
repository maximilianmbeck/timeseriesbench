# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from dataclasses import dataclass
from typing import Optional

import torch

from ..base import LayerConfigInterface, LayerInterface
from ..init import ParameterInit, ParameterInitConfig


@dataclass
class LinearInitConfig:
    weight: Optional[ParameterInitConfig] = None
    bias: Optional[ParameterInitConfig] = None


@dataclass
class LinearConfig(LayerConfigInterface):
    in_features: int = 0
    out_features: int = 0
    bias: bool = True
    init: Optional[LinearInitConfig] = None
    trainable_weight: bool = True
    trainable_bias: bool = True

    def assign_model_config_params(self, model_config):
        pass


class Linear(torch.nn.Linear, LayerInterface):
    config_class = LinearConfig

    def __init__(self, config: LinearConfig):
        self.config = config
        super().__init__(in_features=config.in_features, out_features=config.out_features, bias=config.bias)
        self.weight.requires_grad_(config.trainable_weight)
        if self.bias is not None:
            self.bias.requires_grad_(config.trainable_bias)
        self.reset_parameters()

    def reset_parameters(self, **kwargs):
        super().reset_parameters()
        if self.config.init is not None:
            init_weight = self.config.init.weight
            init_bias = self.config.init.bias

            if init_weight is not None:
                _init_weight = ParameterInit(init_weight)
                _init_weight(self.weight)
            if init_bias is not None and self.bias is not None:
                _init_bias = ParameterInit(init_bias)
                _init_bias(self.bias)

    def _create_weight_decay_optim_groups(self) -> tuple[set[torch.nn.Parameter], set[torch.nn.Parameter]]:
        weight_decay = (self.weight,) if self.config.trainable_weight else ()
        no_weight_decay = ()
        if self.bias is not None:
            no_weight_decay += (self.bias,) if self.config.trainable_bias else ()

        return weight_decay, no_weight_decay
