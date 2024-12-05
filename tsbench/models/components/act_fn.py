# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Type

import torch
from torch import nn

from ..base import LayerConfigInterface, LayerInterface
from ..utils import create_layer


def log_sigmoid(x):
    return torch.where(x > 0.0, torch.log(torch.sigmoid(x)), x - torch.log(1.0 + torch.exp(x)))


class SwishBeta(LayerInterface):
    """Swish activation function with a trainable beta parameter.

    # TODO maxbeck: In an experiment varying trainable_beta did not show any difference in the learning curve.
    # Is this a bug? Needs further investigation.
    """

    def __init__(self, ndim: int = 0, beta_init: float = 1.0, trainable_beta: bool = False):
        super().__init__()
        self.ndim, self.beta_init, self.trainable_beta = ndim, beta_init, trainable_beta
        if trainable_beta or beta_init != 1.0:
            self.beta = nn.Parameter(torch.ones(ndim) * beta_init, requires_grad=trainable_beta)
        else:
            self.beta = None

    def reset_parameters(self):
        if self.beta is not None:
            nn.init.constant_(self.beta, self.beta_init)

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[set[nn.Parameter], set[nn.Parameter]]:
        weight_decay = ()
        no_weight_decay = (self.beta,) if self.beta is not None else ()
        return weight_decay, no_weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.beta is not None:
            return x * torch.sigmoid(self.beta * x)
        else:
            return x * torch.sigmoid(x)

    def __repr__(self):
        return f"SwishBeta(ndim={self.ndim}, beta_init={self.beta_init}, trainable_beta={self.trainable_beta})"


@dataclass
class SwishConfig(LayerConfigInterface):
    beta_init: float = 1.0
    trainable_beta: bool = False
    ndim: int = 0

    def assign_model_config_params(self, model_config):
        pass


class SwishBetaLayer(SwishBeta):
    config_class = SwishConfig

    def __init__(self, config: SwishConfig):
        super().__init__(ndim=config.ndim, beta_init=config.beta_init, trainable_beta=config.trainable_beta)


class ActFnWrapper(LayerInterface):
    def __init__(self, act_fn: Callable[[torch.Tensor], torch.Tensor], acf_fn_name: str = None):
        super().__init__()
        self.act_fn = act_fn
        self.act_fn_name = acf_fn_name if acf_fn_name is not None else act_fn.__name__

    def reset_parameters(self, **kwargs):
        pass

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[set[nn.Parameter], set[nn.Parameter]]:
        return (), ()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.act_fn(x)

    def __repr__(self):
        return f"ActFnWrapper({self.act_fn_name})"


_act_fn_with_params_cls_registry = {
    "swishbeta": SwishBetaLayer,
}

_act_fn_cls_registry = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "identity": nn.Identity,
    "sigmoid": nn.Sigmoid,
    "exp": lambda: torch.exp,
    "swish": nn.SiLU,
}

_act_fn_registry = {
    "gelu": nn.functional.gelu,
    "relu": nn.functional.relu,
    "relu^2": lambda x: torch.square(nn.functional.relu(x)),
    "leakyrelu": nn.functional.leaky_relu,
    "tanh": nn.functional.tanh,
    "identity": lambda x: x,
    "elu": nn.functional.elu,
    "elu+1": lambda x: nn.functional.elu(x) + 1.0,
    "exp": torch.exp,
    "sigmoid": nn.functional.sigmoid,
    "swish": nn.functional.silu,
}


def _get_act_fn(act_fn_name: str, registry: Dict[str, Any]):
    """Returns the activation function class given its name.

    Args:
        act_fn_name (str): The name of the activation function.

    Returns:
        nn.Module: The corresponding pytorch module class.
    """
    if act_fn_name in registry:
        return registry[act_fn_name]
    else:
        assert (
            False
        ), f'Unknown activation function name "{act_fn_name}". Available activation functions are: {str(_act_fn_cls_registry.keys())}'


def get_act_fn(act_fn_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    return _get_act_fn(act_fn_name, _act_fn_registry)


def get_act_fn_cls(act_fn_name: str) -> Type[nn.Module]:
    return _get_act_fn(act_fn_name, _act_fn_cls_registry)


def create_act_fn(config, act_fn_cfg_key: str = "act_fn", registry_update: dict[str, Type] = {}) -> LayerInterface:
    """Create an activation function from a config dataclass object."""
    if not hasattr(config, act_fn_cfg_key):
        raise ValueError(f"Config has no {act_fn_cfg_key} attribute. Edit config or check if layer is correctly named.")
    act_fn_registry_update = copy.deepcopy(_act_fn_with_params_cls_registry)
    act_fn_registry_update.update(registry_update)
    # check if the activation function is a parameterized one
    act_fn_cfg_value = getattr(config, act_fn_cfg_key)
    if isinstance(act_fn_cfg_value, str):
        act_fn_name = act_fn_cfg_value
        return ActFnWrapper(act_fn=get_act_fn(act_fn_name), acf_fn_name=act_fn_name)
    else:
        act_fn_name = getattr(config, act_fn_cfg_key).name
    if act_fn_name in act_fn_registry_update:
        return create_layer(config, registry=act_fn_registry_update, layer_cfg_key=act_fn_cfg_key)
    else:
        return ActFnWrapper(act_fn=get_act_fn(act_fn_name), acf_fn_name=act_fn_name)
