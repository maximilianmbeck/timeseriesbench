# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch


def laplace_(tensor: torch.Tensor, loc: float, scale: float) -> torch.Tensor:
    with torch.no_grad():
        tensor.copy_(torch.distributions.laplace.Laplace(loc, scale).sample(tensor.shape))
    return tensor


def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


@dataclass
class ParameterInitConfig:
    range: float = 0.02
    method: str = "normal"  # "normal", "uniform", "sparse", "orthogonal", "laplace", "zeros", "noop", "constant"
    sparsity: float = 0.0  # 0.0 means dense, 1.0 means all zeros
    start: Optional[float] = None
    end: Optional[float] = None


def init_parameter(param: torch.Tensor, config: ParameterInitConfig) -> None:
    if config.method == "normal":
        torch.nn.init.normal_(param, mean=0.0, std=config.range)
    elif config.method == "uniform":
        torch.nn.init.uniform_(param, a=-config.range, b=config.range)
    elif config.method == "orthogonal":
        torch.nn.init.orthogonal_(param, gain=config.range)
    elif config.method == "sparse":
        torch.nn.init.sparse_(param, sparsity=config.sparsity, std=config.range)
    elif config.method == "laplace":
        laplace_(param, loc=0.0, scale=config.range)
    elif config.method == "zeros":
        torch.nn.init.zeros_(param)
    elif config.method == "constant":
        torch.nn.init.constant_(param, val=config.range)
    elif config.method == "xavier_uniform":
        torch.nn.init.xavier_uniform_(param, gain=config.range)
    elif config.method == "xavier_normal":
        torch.nn.init.xavier_normal_(param, gain=config.range)
    elif config.method == "bias_linspace":
        bias_linspace_init_(param, start=config.start, end=config.end)
    elif config.method == "noop":
        pass
    else:
        raise ValueError(f"Unknown init_embedding_method: {config.method}")


class ParameterInit:
    config_class = ParameterInitConfig

    def __init__(self, config: ParameterInitConfig):
        self.config = config

    def __call__(self, param: torch.Tensor) -> None:
        if param is not None:
            init_parameter(param, config=self.config)


@dataclass
class BaseInitFuncConfig:
    # will be filled from model config
    hidden_size: int = -1
    num_blocks: int = -1
    block_idx: int = -1

    def assign_config_params(self, config, block_idx: int, num_blocks: int):
        if hasattr(config, "head_dim"):
            self.hidden_size = config.head_dim
        else:
            self.hidden_size = config.hidden_size
        self.num_blocks = num_blocks
        self.block_idx = block_idx


### bias init functions


@dataclass
class BiasPowerInitConfig(BaseInitFuncConfig):
    right_x: float = 5.0
    range_x_neg_dir: float = 8.0
    spread_lower: float = 0.7
    spread_upper: float = 1.3


class BiasPowerInit:
    config_class = BiasPowerInitConfig

    def __init__(self, config: BiasPowerInitConfig):
        self.config = config

    def __call__(self, bias: torch.Tensor, gate_idx: int) -> torch.Tensor:
        config = self.config

        def init_func(
            h, hidden_dim, block_idx, num_blocks, right_x=5.0, range_x_neg_dir=8.0, spread_lower=0.7, spread_upper=1.3
        ):
            ratio_0_to_1 = block_idx / (num_blocks - 1) if num_blocks > 1 else 0.0
            init_value = -(  #! minus sign
                -right_x + range_x_neg_dir * ((h) / (hidden_dim - 1)) ** (spread_lower + spread_upper * ratio_0_to_1)
            )
            return init_value

        init_fn = partial(
            init_func,
            hidden_dim=config.hidden_size,
            block_idx=config.block_idx,
            num_blocks=config.num_blocks,
            right_x=config.right_x,
            range_x_neg_dir=config.range_x_neg_dir,
            spread_lower=config.spread_lower,
            spread_upper=config.spread_upper,
        )
        init_vec = torch.arange(config.hidden_size, dtype=bias.dtype, device=bias.device).apply_(init_fn)
        with torch.no_grad():
            bias[gate_idx * config.hidden_size : (gate_idx + 1) * config.hidden_size] = init_vec
