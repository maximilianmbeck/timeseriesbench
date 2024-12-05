# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import math
from dataclasses import dataclass, field

from ..base import LayerConfigInterface
from ..components.linear import LinearInitConfig, ParameterInitConfig
from ..transformer.base import BaseTransformerConfig


@dataclass
class BaseFeedForwardConfig(LayerConfigInterface):
    act_fn: str = "gelu"

    proj_up_init: LinearInitConfig = field(
        default_factory=lambda: LinearInitConfig(
            weight=ParameterInitConfig(method="normal", range=0.02), bias=ParameterInitConfig(method="zeros")
        )
    )
    proj_down_init: LinearInitConfig = field(
        default_factory=lambda: LinearInitConfig(
            weight=ParameterInitConfig(method="normal", range=0.02), bias=ParameterInitConfig(method="zeros")
        )
    )

    # apply std initialization scaling with 1 / math.sqrt(2 * n_layer)) to proj_down layer
    apply_std_init_scaling_downproj: bool = True

    # will be assigned from base model config
    embedding_dim: int = None
    dropout: float = None
    bias: bool = True

    # internal
    _downproj_scale_factor_applied: bool = False

    def assign_model_config_params(self, model_config: BaseTransformerConfig):
        self.embedding_dim = model_config.embedding_dim
        self.dropout = model_config.dropout
        self.bias = model_config.bias
        self.set_std_init_scaling_downproj(num_blocks=model_config.num_layers)

    def set_std_init_scaling_downproj(self, num_blocks: int):
        # in nanogpt.py the c_proj layer is initialized with std 0.02 / math.sqrt(2 * n_layer))
        if self.apply_std_init_scaling_downproj and not self._downproj_scale_factor_applied:
            assert num_blocks > 0, "num_blocks must be > 0"
            self.proj_down_init.weight.range /= math.sqrt(2 * num_blocks)
            self._downproj_scale_factor_applied = True
