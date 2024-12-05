# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Andreas Auer

from typing import Any, Callable, Dict, Tuple, Type, TypeVar

from dacite import Config, from_dict

from .config import NameAndKwargs

factory_type = TypeVar("T")


def get_and_create_class_factory(
    registry: Dict[str, Type[factory_type]], registry_name=None
) -> Tuple[Callable[[str], factory_type], Callable[[str, Dict[str, Any]], factory_type]]:
    registry_name = registry_name if registry_name is not None else f"{registry=}".split("=")[0]

    def get_class(name: str) -> Type[factory_type]:
        if name in registry:
            return registry[name]
        else:
            assert (
                False
            ), f'Unknown {registry_name} name "{name}". Available {registry_name} are: {str(registry.keys())}'

    def create_class(named_kwargs: NameAndKwargs) -> factory_type:
        clazz = get_class(named_kwargs.name)
        config_class = clazz.config_class if hasattr(clazz, "config_class") else None
        if config_class is None:
            return clazz(**named_kwargs.kwargs)
        else:
            config = from_dict(
                data_class=config_class,
                data=named_kwargs.kwargs,
                config=Config(strict=False, strict_unions_match=False),  # TODO Check why strict does not work!!
            )
            initialized = clazz(config)
            return initialized

    return get_class, create_class
