# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from ..base import LayerConfigInterface, LayerInterface


class LayerNorm(LayerInterface):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool = True, weight: bool = True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def reset_parameters(self, **kwargs):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[set[nn.Parameter], set[nn.Parameter]]:
        no_weight_decay = ()
        if self.weight is not None:
            no_weight_decay += (self.weight,)
        if self.bias is not None:
            no_weight_decay += (self.bias,)
        return (), no_weight_decay

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, normalized_shape=self.weight.shape, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSNorm(LayerInterface):
    """RMSNorm with optional bias according to https://arxiv.org/abs/1910.07467.
    Inspired by https://github.com/mistralai/mistral-src/blob/main/mistral/model.py.
    """

    def __init__(self, ndim: int, bias: bool = True, weight: bool = True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.use_weight = weight
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def reset_parameters(self, **kwargs):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[set[nn.Parameter], set[nn.Parameter]]:
        no_weight_decay = ()
        if self.weight is not None:
            no_weight_decay += (self.weight,)
        if self.bias is not None:
            no_weight_decay += (self.bias,)
        return (), no_weight_decay

    def _rms_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, H)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self._rms_normalize(input)
        if self.use_weight and self.use_bias:
            return x * self.weight + self.bias
        elif self.use_weight and not self.use_bias:
            return x * self.weight
        elif not self.use_weight and not self.use_bias:
            return x
        else:
            raise ValueError("RMSNorm: combination use_weight=False and use_bias=True not possible.")


class MultiHeadLayerNorm(LayerInterface):
    """MultiHead Layernorm but with an optional bias. PyTorch doesn't support simply bias=False.
    # TODO add better doc.
    Implemented after the description in the Retention paper Section 2.2 .
    Args:
        ndim (int): number of dimensions of the input, will be separated into num_heads groups
    """

    def __init__(self, ndim: int, num_heads: int, bias: bool = True, weight: bool = True):
        super().__init__()
        param_dim = ndim  # we want to have a separate bias/weight for each channel
        self.weight = nn.Parameter(torch.ones(param_dim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(param_dim)) if bias else None
        self.num_heads = num_heads

    def reset_parameters(self, **kwargs):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _create_weight_decay_optim_groups(
        self,
    ) -> tuple[set[nn.Parameter], set[nn.Parameter]]:
        no_weight_decay = ()
        if self.weight is not None:
            no_weight_decay += (self.weight,)
        if self.bias is not None:
            no_weight_decay += (self.bias,)
        return (), no_weight_decay

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (B, NH, S, DH)
        batch_size, num_heads, seq_len, head_dim = input.shape
        # gn_in = rearrange(input, 'B NH S DH -> (B S) (NH DH)')
        gn_in = input.transpose(1, 2)  # (B, NH, S, DH) -> (B, S, NH, DH)
        gn_in = gn_in.reshape(batch_size * seq_len, num_heads * head_dim)  # (B, S, NH, DH) -> (B * S), (NH * DH)
        out = F.group_norm(gn_in, self.num_heads, weight=self.weight, bias=self.bias, eps=1e-5)

        # out = rearrange(out, '(B S) (NH DH) -> B NH S DH', B=batch_size, S=seq_len, NH=num_heads, DH=head_dim)
        out = out.view(batch_size, seq_len, num_heads, head_dim).transpose(
            1, 2
        )  # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        return out


@dataclass
class NormConfig(LayerConfigInterface):
    norm_type: str = "layer"  # "rms" or "layer"
    ndim: Optional[int] = None
    bias: bool = True
    weight: bool = True
    eps: float = 1e-5

    def assign_model_config_params(self, model_config):
        pass


def create_norm_layer(config: NormConfig, ndim: int = None) -> LayerInterface:
    _ndim = ndim
    if _ndim is None:
        _ndim = config.ndim

    assert _ndim is not None, "ndim must be set for norm layer creation"

    if config.norm_type == "rms":
        return RMSNorm(ndim=_ndim, bias=config.bias, weight=config.weight, eps=config.eps)
    elif config.norm_type == "layer":
        return LayerNorm(ndim=_ndim, bias=config.bias, weight=config.weight, eps=config.eps)
    else:
        raise ValueError(f"Unknown norm_type {config.norm_type}")
