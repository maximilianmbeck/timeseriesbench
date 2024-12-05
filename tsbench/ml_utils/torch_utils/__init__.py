import logging
from typing import Iterable, Union

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


def padded_stack(
    tensors: list[torch.Tensor], side: str = "right", mode: str = "constant", value: Union[int, float] = 0
) -> torch.Tensor:
    """
    Stack tensors along first dimension and pad them along last dimension to ensure their size is equal.
    Copy from https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting/utils.html#padded_stack
    Args:
        tensors (list[torch.Tensor]): list of tensors to stack
        side (str): side on which to pad - "left" or "right". Defaults to "right".
        mode (str): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        value (Union[int, float]): value to use for constant padding

    Returns:
        torch.Tensor: stacked tensor
    """
    full_size = max([x.size(-1) for x in tensors])

    def make_padding(pad):
        if side == "left":
            return (pad, 0)
        elif side == "right":
            return (0, pad)
        else:
            raise ValueError(f"side for padding '{side}' is unknown")

    out = torch.stack(
        [
            F.pad(x, make_padding(full_size - x.size(-1)), mode=mode, value=value) if full_size - x.size(-1) > 0 else x
            for x in tensors
        ],
        dim=0,
    )
    return out


def gradients_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    """Convert gradients to one vector.
    The returned tensor is a copy of the parameters.

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameter gradients represented by a single vector (copy of the parameters)
    """
    from torch.nn.utils.convert_parameters import _check_param_device

    # Flag for the device where the parameter is located
    param_device = None

    grad = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)
        if param.grad is None:
            # return None
            continue
        grad.append(param.grad.view(-1))
    if grad == []:
        return None
    return torch.cat(grad)


def cummean(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute the cumulative mean along a given dimension.

    Args:
        x (torch.Tensor): the input tensor.
        dim (int, optional): The dimension to average over. Defaults to -1.

    Returns:
        torch.Tensor: the cumulative mean.
    """
    sizes = [
        1,
    ] * len(x.shape)
    sizes[dim] = -1
    return x.cumsum(dim=dim) / (torch.arange(x.size(dim), device=x.device) + 1.0).view(sizes)


def set_seed(seed: int) -> None:
    import os
    import random

    import numpy as np

    # Additionally, some operations on a GPU are implemented stochastic for
    # efficiency
    # We want to ensure that all operations are deterministic on GPU (if used)
    # for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # For `use_deterministic_algorithms` need to set CUBLAS_WORKSPACE_CONFIG
    # environment variable when using CUDA version 10.2 or greater
    # https://pytorch.org/docs/stable/notes/randomness#avoiding-nondeterministic-algorithms
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(":4096:8")
    torch.use_deterministic_algorithms(mode=False, warn_only=False)

    np.random.seed(seed)  # only sets seed for old API
    # take care when using the new API, i.e. default_rng()!
    # default_rng(seed=None) always pulls a fresh, unpredictable entropy from
    # the OS
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device: Union[torch.device, str, int]) -> torch.device:
    if device == "auto":
        device = "cuda"
    if isinstance(device, int):
        if device < 0:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{device}")
    else:
        device = torch.device(device)

    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        LOGGER.warn(f"Device '{str(device)}' is not available! Using cpu now.")
        return torch.device("cpu")
    return device
