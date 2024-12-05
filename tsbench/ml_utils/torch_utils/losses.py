# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from abc import ABC
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..meta_factory import get_and_create_class_factory


class BaseLoss(ABC):
    def _prechecks(self, logits: torch.Tensor, targets: torch.Tensor):
        assert not torch.any(torch.isnan(logits.view(-1)))
        assert not torch.any(torch.isnan(targets.view(-1)))


@dataclass
class CrossEntropyLossConfig:
    ignore_index: int = -1


class CrossEntropyLossSequence(BaseLoss):
    """Cross entropy loss for sequence classification tasks.
    Assumes that the input is a sequence of logits."""

    config_class = CrossEntropyLossConfig

    def __init__(self, config: CrossEntropyLossConfig):
        self.config = config

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor):
        self._prechecks(logits=logits, targets=targets)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.config.ignore_index
        )
        return loss


# TODO add MSE loss

_loss_registry = {
    "crossentropy_sequence": CrossEntropyLossSequence,
    "crossentropy_torch": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
}

get_loss, create_loss = get_and_create_class_factory(registry=_loss_registry, registry_name="loss")
