# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import copy
from typing import Callable, Mapping, Sequence, Union

import torch
from torch import nn

from ..ddp_utils.primitives import all_gather_tensor_list
from ..ddp_utils.setup import is_distributed


def move_to(
    obj: Union[torch.Tensor, Sequence, Mapping], device: torch.device
) -> Union[torch.Tensor, list, dict, tuple]:
    """Moves all tensors in `obj` to `device`."""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, Mapping):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, Sequence):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError(f"Invalid type `{type(obj)}` for move_to.")


def unwrap_model(model: nn.Module) -> nn.Module:
    # this model should return a pointer to the model, therefore calling the state_dict() method should work
    # unwrap model from DDP
    temp_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    # unwrap model from compiled model
    temp_model = temp_model._orig_mod if hasattr(temp_model, "_orig_mod") else temp_model
    return temp_model


class DataBatch:
    """This is a wrapper around a batch of data received from a DataLoader.
    It is used to pass data to the model during training and inference.
    Sometimes the dataset returns a tuple of tensors, sometimes a dict of tensors.
    This class is used to wrap the data and provide a common interface for the trainer.
    """

    def __init__(self, batch: Sequence[torch.Tensor | dict[str, torch.Tensor]]):
        assert isinstance(batch, Sequence), f"Invalid type `{type(batch)}` for batch. Must be a Sequence."
        assert len(batch) > 0, "Batch must not be empty."
        assert len(batch) <= 3, "Batch must not contain more than 3 elements."
        self._batch = batch

    def move_to(self, device: torch.device) -> "DataBatch":
        """Moves the data to the specified device."""
        self._batch = move_to(self._batch, device)
        return self

    @property
    def xs(self) -> torch.Tensor:
        """Returns the input data."""
        return self._batch[0]

    @property
    def ys(self) -> torch.Tensor:
        """Returns the target data."""
        return self._batch[1]

    @property
    def meta(self) -> dict[str, torch.Tensor]:
        """Returns the meta data."""
        if len(self._batch) > 2:
            return self._batch[2]
        else:
            return {}


class ValueAccumulator:
    """This class is used to accumulate values over multiple steps.
    It operates on dictionaries and allows to add values to the dictionary.
    It also synchronizes the values across multiple processes, upon reducing.

    Currently only torch.Tensors are supported as values.

    It has the following capabilities:
    - add values to the accumulator:
        - only accepts keys that are already in the accumulator
    - reset the accumulator:
        - reset all keys empty lists, i.e. clear all values (but keep the keys)
    - reduce the values:
        - all gather the values across all processes (if distributed and configured to do so)
        - reduce the values to a single value (e.g. mean, sum, etc.)
        - if no values are present, return nan
    """

    single_val_key = "_single_val_key"
    reduce_fn_registry = {"mean": torch.mean, "sum": torch.sum, "max": torch.max, "min": torch.min}

    def __init__(
        self,
        values: dict[str, torch.Tensor] | torch.Tensor = {},
        allow_add_empty_dict: bool = False,
    ):
        self.allow_add_empty_dict = allow_add_empty_dict
        self._values = dict()
        self._initialized = False
        input_values = self._handle_input_values(values)
        if len(input_values) > 0:
            self._initialize_values(input_values)

    def _handle_input_values(self, values: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor]:
        """Bring input values into a dict-like format `dict[str, torch.Tensor]`.
        Detach tensors from the graph. We never propagate gradients through the accumulator.
        """
        if isinstance(values, torch.Tensor):
            return {self.single_val_key: values.detach()}
        elif isinstance(values, dict):
            for key, value in values.items():
                assert isinstance(
                    value, torch.Tensor
                ), f"Invalid type `{type(value)}` for value. Must be a torch.Tensor."
                values[key] = value.detach()
            return values
        else:
            raise ValueError(f"Invalid type `{type(values)}` for values. Accept only dict or torch.Tensor.")

    def _initialize_values(self, values: dict[str, torch.Tensor]) -> None:
        assert len(values) > 0, "Values must not be empty."
        for key, value in values.items():
            assert isinstance(value, torch.Tensor), f"Invalid type `{type(value)}` for value. Must be a torch.Tensor."
            self._values[key] = [value]
        self._initialized = True

    def add(self, values: dict[str, torch.Tensor] | torch.Tensor) -> None:
        values = self._handle_input_values(values)

        if not self._initialized:
            self._initialize_values(values)
        else:
            self._add_values(values)

    def _add_values(self, values: dict[str, torch.Tensor]) -> None:
        for key, value in values.items():
            assert isinstance(value, torch.Tensor), f"Invalid type `{type(value)}` for value. Must be a torch.Tensor."
            if key not in self._values:
                raise KeyError(f"Key `{key}` not in accumulator. For new keys create a new Accumulator object.")
            self._values[key].append(value)

    def reset(self) -> None:
        """Clears all values in the accumulator, but keeps the keys."""
        for key in self._values.keys():
            self._values[key] = []

    @torch.no_grad()
    def reduce(
        self,
        reduce_fn: str | Callable[[list[torch.Tensor]], torch.Tensor] = "mean",
        dist_sync_before_reduce: bool = True,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Reduces the values across all processes and returns the reduced values.
        This function makes a copy of the current vales and does not clear the values in the accumulator.
        In order to clear the values, call `reset()`. To clear the keys create a new Accumulator object."""

        if not self._initialized:
            raise RuntimeError("Cannot reduce empty, non-initialized accumulator.")

        if dist_sync_before_reduce and is_distributed():
            vals_to_reduce = self._dist_sync_values()
        else:
            vals_to_reduce = copy.deepcopy(self._values)

        reduce_fn = self._get_reduce_fn(reduce_fn)
        reduced_values = self._reduce_values(vals_to_reduce, reduce_fn)

        if len(reduced_values) == 1 and self.single_val_key in reduced_values:
            return reduced_values[self.single_val_key]
        else:
            return reduced_values

    def _get_reduce_fn(
        self, reduce_fn: str | Callable[[list[torch.Tensor]], torch.Tensor]
    ) -> Callable[[list[torch.Tensor]], torch.Tensor]:
        if isinstance(reduce_fn, str):
            torch_reduce_fn = self.reduce_fn_registry.get(reduce_fn, None)
            if torch_reduce_fn is None:
                raise ValueError(
                    f"Invalid value `{reduce_fn}` for reduce_fn. Available reduce functions: {self.reduce_fn_registry.keys()}"
                )
            return lambda x: torch_reduce_fn(torch.stack(x))
        elif callable(reduce_fn):
            return reduce_fn
        else:
            raise ValueError(f"Invalid type `{type(reduce_fn)}` for reduce_fn.")

    def _dist_sync_values(self) -> dict[str, list[torch.Tensor]]:
        """Synchronizes the values across all processes."""
        values = copy.deepcopy(self._values)  # do not modify original values
        # values are a dictioniary of lists of tensors
        # we need to synchronize the lists of tensors across all processes
        # we do this by stacking the tensors in the lists and then all_gather the stacked tensors
        for key, value_list in values.items():
            # do not modify values if there is no value
            if len(value_list) > 0:
                value_list = all_gather_tensor_list(value_list)
                values[key] = value_list

        return values

    def _reduce_values(
        self, values: dict[str, list[torch.Tensor]], reduce_fn: Callable[[list[torch.Tensor]], torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        reduced_values = {}
        for key, value_list in values.items():
            if len(value_list) > 0:
                reduced_values[key] = reduce_fn(value_list)
            else:
                reduced_values[key] = torch.tensor(float("nan"))
        return reduced_values
