# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from ..interfaces import SequenceConfigInterface, SequenceInterface
from ..ml_utils.models.base_model import BaseModel

LOGGER = logging.getLogger(__name__)


class LayerConfigInterface(ABC):
    @abstractmethod
    def assign_model_config_params(self, model_config):
        pass


@dataclass
class BaseSequenceModelConfig(SequenceConfigInterface):
    shortname: str = ""  # needed to give a model a more distinctive name used in configurations etc., temporary filled by hydra or OmegaConf


class ResettableParametersModule(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def reset_parameters(self, **kwargs):
        pass


class WeightDecayOptimGroupInterface(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        """Return a tuple of two sequences, one for parameters with weight decay and one for parameters without weight decay.
        Performs checks to ensure that each parameter is only in one of the two sequences.
        """
        weight_decay, no_weight_decay = self._create_weight_decay_optim_groups(**kwargs)

        # Check that parameters have been assigned correctly.
        # Each parameter can only be in one optim group.
        intersection_params = set(weight_decay).intersection(set(no_weight_decay))
        assert (
            len(intersection_params) == 0
        ), f"parameters {[pn for pn, p in self.named_parameters() if p in intersection_params]} made it into both decay/no_decay sets!"

        union_params = set(weight_decay).union(set(no_weight_decay))
        param_dict = {pn: p for pn, p in self.named_parameters()}
        unassigned_params = set(param_dict.values()) - union_params
        # We have parameters that were not assigned to either weight decay or no weight decay.
        # Find the parameter names and raise an error.
        assert (
            len(unassigned_params) == 0
        ), f"Parameters {[pn for pn, p in self.named_parameters() if p in unassigned_params]} were not separated into either decay/no_decay set!"

        return weight_decay, no_weight_decay

    def get_weight_decay_optim_group_param_names(self, **kwargs) -> tuple[Sequence[str], Sequence[str]]:
        """Return a tuple of two sequences, one for parameter names with weight decay and one for parameter names without weight decay.
        Performs checks to ensure that each parameter is only in one of the two sequences.
        """

        def _is_in_sequence(param: nn.Parameter, sequence: Sequence[nn.Parameter]) -> bool:
            for p in sequence:
                if param is p:
                    return True
            return False

        weight_decay, no_weight_decay = self.get_weight_decay_optim_groups(**kwargs)
        names_weight_decay = [pn for pn, p in self.named_parameters() if _is_in_sequence(p, weight_decay)]
        names_no_weight_decay = [pn for pn, p in self.named_parameters() if _is_in_sequence(p, no_weight_decay)]
        return names_weight_decay, names_no_weight_decay

    def _create_weight_decay_optim_groups(
        self, normalization_weight_decay: bool = False, **kwargs
    ) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        """Return a tuple of two sequences, one for parameters with weight decay and one for parameters without weight decay.
        Default separation:
        - weight decay: nn.Linear
        - no weight decay: nn.LayerNorm, nn.Embedding, nn.BatchNormXd, nn.GroupNorm"""
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.Embedding,)
        blacklist_names = ("embedding", "Embedding")
        if not normalization_weight_decay:
            blacklist_weight_modules += (
                nn.LayerNorm,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.GroupNorm,
            )

        decay = set()
        no_decay = set()
        for name, module in self.named_modules():
            if isinstance(module, whitelist_weight_modules):
                for pn, p in module.named_parameters():
                    if any(bl_name in name + "." + pn for bl_name in blacklist_names):
                        no_decay.add(p)
                    elif pn.endswith("weight"):
                        decay.add(p)
                    elif pn.endswith("bias"):
                        # biases will not experience regular weight decay
                        no_decay.add(p)
                    else:
                        decay.add(p)
            elif isinstance(module, blacklist_weight_modules):
                no_decay.update(set(module.parameters()))

        return tuple(decay), tuple(no_decay)

    def _get_weight_decay_optim_groups_for_modules(
        self, modules: list["WeightDecayOptimGroupInterface"], **kwargs
    ) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        weight_decay, no_weight_decay = (), ()
        for module in modules:
            wd, nwd = module.get_weight_decay_optim_groups(**kwargs)
            weight_decay += wd
            no_weight_decay += nwd
        return weight_decay, no_weight_decay


class LayerInterface(ResettableParametersModule, WeightDecayOptimGroupInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BaseSequenceModelTrain(BaseModel, LayerInterface, SequenceInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure_optim_groups(self, optim_cfg: dict) -> torch.optim.Optimizer:
        weight_decay, no_weight_decay = self.get_weight_decay_optim_groups()

        weight_decay_names, no_weight_decay_names = self.get_weight_decay_optim_group_param_names()
        LOGGER.info(f"Weight decay applied to: {sorted(list(weight_decay_names))}")
        LOGGER.info(f"No weight decay applied to: {sorted(list(no_weight_decay_names))}")

        optim_groups = [
            {"params": list(weight_decay), "weight_decay": optim_cfg["weight_decay"]},
            {"params": list(no_weight_decay), "weight_decay": 0.0},
        ]

        return optim_groups
