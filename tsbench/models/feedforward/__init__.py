# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from functools import partial

from ..utils import create_layer
from .ffn import FeedForward
from .ffn_gate import FeedForwardGate

_feedforward_registry = {
    "ff": FeedForward,
    "ff_gate": FeedForwardGate,
}


create_feedforward_layer = partial(create_layer, registry=_feedforward_registry, layer_cfg_key="feedforward")
