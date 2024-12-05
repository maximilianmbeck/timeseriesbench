from functools import partial

from ..utils import create_layer
from .attention_layers import CausalSelfAttention
from .lstms import LSTM

_sequence_mix_registry = {
    "causalselfattention": CausalSelfAttention,
    "lstm": LSTM,
}


create_sequence_mix_layer = partial(create_layer, registry=_sequence_mix_registry, layer_cfg_key="sequence_mix")
