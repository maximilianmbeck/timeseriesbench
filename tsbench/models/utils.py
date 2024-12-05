# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from typing import Any, Type

from dacite import Config, from_dict

from ..ml_utils.config import NameAndKwargs
from .base import LayerInterface


# TODO what would be a typehint for a generic dataclass?
def create_layer(
    config, registry: dict[str, Type], layer_cfg_key: str, nameandkwargs: NameAndKwargs = None, **kwargs
) -> LayerInterface:
    """Create a layer from a config dataclass object, a layer class registry and a layer config key.
    The layer config key is the name of the attribute in the config object that contains the layer config.

    This function assumes that the config object is a hierarchical dataclass object, i.e. it contains configurable
    layers of type `NameAndKwargs` (which is a dataclass with keys `name` and `kwargs`).
    The `name` attribute is used to get the layer class from the registry and the `kwargs` attribute is used to
    instantiate the layer class.

    If `nameandkwargs` is not None, it is used instead of the `NameAndKwargs` attribute in the config object.

    If the layer class has a `config_class` attribute, the `kwargs` attribute is used to instantiate the layer config.

    Args:
        config (dataclass): config object created
        registry (dict[str, Type]): layer class registry
        layer_cfg_key (str): layer config key
        nameandkwargs (NameAndKwargs, optional): layer name and kwargs. Defaults to None.

    Returns:
        LayerInterface: layer instance
    """

    def get_layer(name: str) -> Type:
        if name in registry:
            return registry[name]
        else:
            raise ValueError(
                f"Unknown {layer_cfg_key} layer: {name}. Available {layer_cfg_key} layers: {list(registry.keys())}"
            )

    if nameandkwargs is not None:
        cfg_name = nameandkwargs.name
        cfg_kwargs = nameandkwargs.kwargs
    else:
        if not hasattr(config, layer_cfg_key):
            raise ValueError(
                f"Config {config} has no {layer_cfg_key} attribute. Edit config or check if layer is correctly named."
            )
        cfg_name = getattr(config, layer_cfg_key).name
        cfg_kwargs = getattr(config, layer_cfg_key).kwargs

    layer_class = get_layer(cfg_name)
    layer_config_class = getattr(layer_class, "config_class", None)
    if layer_config_class is None:
        assert not kwargs, (
            f"Layer {cfg_name} does not have a config class ",
            "but other kwargs is not None.",
        )
        return layer_class(**cfg_kwargs) if cfg_kwargs is not None else layer_class()
    else:
        layer_config = from_dict(
            data_class=layer_config_class,
            data=(cfg_kwargs if cfg_kwargs is not None else {}),
            config=Config(strict=True, strict_unions_match=True),
        )
        layer_config.assign_model_config_params(model_config=config)
        if kwargs:
            layer = layer_class(layer_config, **kwargs)
        else:
            layer = layer_class(layer_config)
        return layer
