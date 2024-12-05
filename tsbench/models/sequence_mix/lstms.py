# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from torch.nn import LSTM as _TorchLSTM

from ..base import LayerConfigInterface, LayerInterface
from ..init import BiasPowerInit, BiasPowerInitConfig
from ..transformer.base import BaseTransformerConfig


@dataclass
class LSTMConfig(LayerConfigInterface):
    type: str = "torch"  # options: torch
    backend: str = "vanilla"  # options: vanilla, fwbw, cuda (for type!=torch only)
    init: Optional[BiasPowerInitConfig] = None

    # will be assigned from base model config
    input_size: int = None
    hidden_size: int = None

    def assign_model_config_params(self, model_config: BaseTransformerConfig):
        self.input_size = model_config.embedding_dim
        self.hidden_size = model_config.embedding_dim


class LSTM(LayerInterface):
    config_class = LSTMConfig

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        if config.type == "torch":
            self._internal_config = None
            self.lstm = _TorchLSTM(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
        else:
            raise ValueError(f"Unknown LSTM type: {config.type}")

    def reset_parameters(self, **kwargs):
        self.lstm.reset_parameters()
        if self.config.init is not None:
            if self.config.type == "torch":
                self.config.init.assign_config_params(self.config, kwargs["block_idx"], kwargs["num_blocks"])
                power_init = BiasPowerInit(self.config.init)
                with torch.no_grad():
                    self.lstm.bias_hh_l0[:] = 0.0 * self.lstm.bias_hh_l0
                power_init(self.lstm.bias_ih_l0, 1)

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[set[nn.Parameter], set[nn.Parameter]]:
        weight_decay, no_weight_decay = (), ()
        for pn, p in self.lstm.named_parameters():
            if "bias" in pn:
                no_weight_decay += (p,)
            else:
                weight_decay += (p,)
        return weight_decay, no_weight_decay

    def forward(self, x: torch.Tensor, return_state: bool = False, **kwargs) -> torch.Tensor:
        out, state = self.lstm(x)
        if return_state:
            return out, state
        return out
