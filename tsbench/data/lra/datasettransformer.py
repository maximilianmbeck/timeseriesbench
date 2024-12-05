# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck


import inspect
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import torch
import torchvision
from omegaconf import DictConfig
from torch.utils.data import Dataset

available_pytorch_visiontransforms = torchvision.transforms.transforms.__all__

__pytorch_visiontransforms = inspect.getmembers(
    torchvision.transforms,
    lambda transform_class: inspect.isclass(transform_class)
    and transform_class.__name__ in available_pytorch_visiontransforms,
)

_pytorch_transforms_registry = {name: cls for name, cls in __pytorch_visiontransforms}

# for now there are no custom transforms
_transforms_registry = _pytorch_transforms_registry


def get_transform_class(transform_name: str) -> Type:
    if transform_name in _transforms_registry:
        return _transforms_registry[transform_name]
    else:
        assert (
            False
        ), f'Unknown transform name "{transform_name}". Available transforms are: {str(_transforms_registry.keys())}'


def create_transform(transform_cfg: Union[str, Dict[str, Any]]) -> Callable:
    if isinstance(transform_cfg, str):
        transform_cls = get_transform_class(transform_cfg)
        return transform_cls()
    elif isinstance(transform_cfg, (dict, DictConfig)):
        transform_name = list(transform_cfg.keys())
        assert len(transform_name) == 1, f"No or multiple transform config passed. Expect only one!"
        transform_name = transform_name[0]
        transform_cls = get_transform_class(transform_name)
        transform_args = transform_cfg[transform_name]
        transform = (
            transform_cls(*transform_args) if isinstance(transform_args, list) else transform_cls(**transform_args)
        )
        return transform
    else:
        raise TypeError


def get_transform_classes(transforms: List[Callable]) -> List[Type]:
    return [t.__class__ for t in transforms]


class DatasetTransformer(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        image_transforms: List[Callable] = [],
        tensor_transforms: List[Callable] = [],
        joint_tensor_transforms: List[Callable] = [],
        enable_transforms: bool = True,
    ):
        self.dataset = dataset

        self._image_transforms = image_transforms if enable_transforms else []
        self._tensor_transforms = tensor_transforms if enable_transforms else []
        self._joint_tensor_transforms = joint_tensor_transforms if enable_transforms else []

        # avoid applying the ToTensor transform twice
        if not tensor_transforms:
            assert torchvision.transforms.ToTensor in get_transform_classes(
                self._image_transforms
            ), "If no tensor transforms are passed, the image transforms must contain the ToTensor transform!"
            self._composed_image_tensor_transforms = torchvision.transforms.Compose(self._image_transforms)
        else:
            self._composed_image_tensor_transforms = torchvision.transforms.Compose(
                self._image_transforms + [torchvision.transforms.ToTensor()] + self._tensor_transforms
            )

    @property
    def image_tensor_transforms(self) -> Callable:
        return self._composed_image_tensor_transforms

    @property
    def joint_tensor_transforms(self) -> List[Callable]:
        return self._joint_tensor_transforms

    def transform(self, input, target=None) -> Tuple[torch.Tensor, torch.Tensor]:
        input = self._composed_image_tensor_transforms(input)

        for joint_tensor_transform in self._joint_tensor_transforms:
            input, target = joint_tensor_transform(input, target)

        return input, target

    def get_raw_item(self, index) -> Tuple[Any, Any]:
        return self.dataset[index]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.dataset[index]
        transformed = self.transform(*raw)
        return transformed

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def create(
        dataset: Dataset,
        image_transforms: Dict[str, Any] = {},
        tensor_transforms: Dict[str, Any] = {},
        joint_tensor_transforms: Dict[str, Any] = {},
        enable_transforms: bool = True,
    ) -> "DatasetTransformer":
        if dataset is None:
            return None

        it = [create_transform(t) for t in image_transforms] if image_transforms else []
        tt = [create_transform(t) for t in tensor_transforms] if tensor_transforms else []
        jtt = [create_transform(t) for t in joint_tensor_transforms] if joint_tensor_transforms else []

        return DatasetTransformer(
            dataset,
            image_transforms=it,
            tensor_transforms=tt,
            joint_tensor_transforms=jtt,
            enable_transforms=enable_transforms,
        )
