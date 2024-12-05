# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck


from dataclasses import dataclass
from typing import Sequence

import torchvision
from einops import rearrange
from torch.utils import data

from ...permutations import bitreversal_permutation
from ..imagesequencedataset import (
    ImageSequenceDatasetConfig,
    ImageSequenceDatasetGenerator,
    Transform,
)


@dataclass
class Cifar10SequenceDatasetConfig(ImageSequenceDatasetConfig):
    tokenize: bool = False
    augment: bool = False
    mode: str = "grayscale"  # 'grayscale', 'rgb', 'rgbrgb', 'rrggbb

    def __post_init__(self):
        assert self.mode in list(
            Cifar10SequenceDatasetGenerator.mode_to_permutation.keys()
        ), f"Unsupported mode: {self.mode}"
        assert not (
            self.tokenize and not self.mode != "rgb"
        ), "Tokenization not supported for 3 channel rgb input, consider using `rgbrgb` or `rrggbb`."


class Cifar10SequenceDatasetGenerator(ImageSequenceDatasetGenerator):
    config_class = Cifar10SequenceDatasetConfig
    # greyscale normalizer values from: https://github.com/HazyResearch/safari/blob/26f6223254c241ec3418a0360a3a704e0a24d73d/src/dataloaders/basic.py#L115
    greyscale_normalizer = {"mean": [122.6 / 255.0], "std": [61.0 / 255.0]}
    # rgb normalizer values from: https://github.com/HazyResearch/safari/blob/26f6223254c241ec3418a0360a3a704e0a24d73d/src/dataloaders/basic.py#L122
    rgb_normalizer = {"mean": [0.4914, 0.4822, 0.4465], "std": [0.247, 0.243, 0.261]}

    img_size = 32
    mode_to_permutation = {
        "grayscale": lambda x: rearrange(x, "c h w -> (h w) c"),
        "rgb": lambda x: rearrange(x, "c h w -> (h w) c"),
        "rgbrgb": lambda x: rearrange(x, "c h w -> (h w c) 1"),
        "rrggbb": lambda x: rearrange(x, "c h w -> (c h w) 1"),
    }

    mode_to_input_dim = {"grayscale": 1, "rgb": 3, "rgbrgb": 1, "rrggbb": 1}

    def __init__(self, config: Cifar10SequenceDatasetConfig):
        super().__init__(config)
        self.config = config

    def _create_image_transforms(self) -> tuple[list[Transform], list[Transform]]:
        # augemntations
        augmentations = []
        if self.config.augment:
            augmentations.extend(
                [
                    torchvision.transforms.RandomCrop(self.img_size, padding=4, padding_mode="symmetric"),
                    torchvision.transforms.RandomHorizontalFlip(),
                ]
            )

        # color
        preprocessors = []
        if self.config.mode == "grayscale":
            preprocessors.append(torchvision.transforms.Grayscale())
        # normalization
        normalizer_values = self.greyscale_normalizer if self.config.mode == "grayscale" else self.rgb_normalizer
        preprocessors.extend([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(**normalizer_values)])

        # permutation
        perm = self.mode_to_permutation[self.config.mode]
        permutations = [torchvision.transforms.Lambda(lambda x: perm(x))]
        # shape: (sequence_length, input_dim)

        if self.config.tokenize:
            # for tokenization we first convert the channel values to integers in range [0, 255]
            # then we reshape the image to a sequence of integers
            assert (
                self.config.mode != "rgb"
            ), "Tokenization not supported for 3 channel rgb input, consider using `rgbrgb` or `rrggbb`."
            preprocessors.append(torchvision.transforms.Lambda(lambda x: (x * 255).long()))
            permutations.append(torchvision.transforms.Lambda(lambda x: rearrange(x, "T 1 -> T")))
            # shape: (sequence_length, )

        train_transforms: list[Transform] = augmentations + preprocessors + permutations
        val_transforms: list[Transform] = preprocessors + permutations
        return train_transforms, val_transforms

    def _load_raw_datasets(self) -> dict[str, data.Dataset]:
        raw_datasets = {}
        raw_datasets["train"] = torchvision.datasets.CIFAR10(root=self.config.data_dir, train=True, download=True)
        raw_datasets["test"] = torchvision.datasets.CIFAR10(root=self.config.data_dir, train=False, download=True)
        return raw_datasets

    @property
    def output_dim(self) -> Sequence[int]:
        return (10,)  # 10 classes

    @property
    def input_dim(self) -> Sequence[int]:
        if self.config.tokenize:
            return None
        return (self.mode_to_input_dim[self.config.mode],)

    @property
    def vocab_size(self) -> int:
        return 256 if self.config.tokenize else None

    @property
    def context_length(self) -> int:
        factor = 1 if self.config.mode in ["grayscale", "rgb"] else 3
        return self.img_size**2 * factor


@dataclass
class MNISTSequenceDatasetConfig(ImageSequenceDatasetConfig):
    tokenize: bool = False
    permutation: str = "none"  # options: bitreversal, none
    normalize: bool = False


class MNISTSequenceDatasetGenerator(ImageSequenceDatasetGenerator):
    config_class = MNISTSequenceDatasetConfig

    img_size = 28
    normalizer = {"mean": [0.1307], "std": [0.3015]}

    def __init__(self, config: MNISTSequenceDatasetConfig):
        super().__init__(config)
        self.config = config

    def _create_image_transforms(self) -> tuple[list[Transform], list[Transform]]:
        preprocessors = [torchvision.transforms.ToTensor()]
        if self.config.normalize:
            preprocessors.append(torchvision.transforms.Normalize(**self.normalizer))
        permutations = [torchvision.transforms.Lambda(lambda x: rearrange(x, "c h w -> (h w) c"))]
        if self.config.permutation == "bitreversal":
            permutation = bitreversal_permutation(self.context_length)
            permutations.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.config.permutation == "none":
            pass
        else:
            raise ValueError(f"Unknown permutation {self.config.permutation}")

        if self.config.tokenize:
            # for tokenization we first convert the channel values to integers in range [0, 255]
            # then we reshape the image to a sequence of integers
            preprocessors.append(torchvision.transforms.Lambda(lambda x: (x * 255).long()))
            permutations.append(torchvision.transforms.Lambda(lambda x: rearrange(x, "T 1 -> T")))
            # shape: (sequence_length, )

        train_transforms = preprocessors + permutations
        val_transforms = preprocessors + permutations

        return train_transforms, val_transforms

    def _load_raw_datasets(self) -> dict[str, data.Dataset]:
        raw_datasets = {}
        raw_datasets["train"] = torchvision.datasets.MNIST(root=self.config.data_dir, train=True, download=True)
        raw_datasets["test"] = torchvision.datasets.MNIST(root=self.config.data_dir, train=False, download=True)
        return raw_datasets

    @property
    def output_dim(self) -> Sequence[int]:
        return (10,)  # 10 classes

    @property
    def input_dim(self) -> Sequence[int]:
        if self.config.tokenize:
            return None
        return (1,)

    @property
    def vocab_size(self) -> int:
        return 256 if self.config.tokenize else None

    @property
    def context_length(self) -> int:
        return self.img_size**2
