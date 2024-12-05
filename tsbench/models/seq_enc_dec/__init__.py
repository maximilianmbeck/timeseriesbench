# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from functools import partial

from ..utils import create_layer
from .decoders import SequenceDecoder
from .encoders import EmbeddingEncoder, LinearEncoder
from .lra_decoders import AanDecoder

_encoder_registry = {"linear": LinearEncoder, "embedding": EmbeddingEncoder}
_decoder_registry = {"sequence": SequenceDecoder, "aanretrieval": AanDecoder}


create_encoder = partial(create_layer, registry=_encoder_registry, layer_cfg_key="encoder")
create_decoder = partial(create_layer, registry=_decoder_registry, layer_cfg_key="decoder")
