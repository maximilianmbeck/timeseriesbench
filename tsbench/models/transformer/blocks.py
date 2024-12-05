# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from dataclasses import dataclass, field
from functools import partial
from typing import Sequence

import torch
from torch import nn

from ...ml_utils.config import NameAndKwargs
from ..base import LayerConfigInterface, LayerInterface
from ..components.ln import NormConfig, create_norm_layer
from ..feedforward import create_feedforward_layer
from ..sequence_mix import create_sequence_mix_layer
from ..utils import create_layer
from .base import BaseTransformerConfig


@dataclass
class PreNormBlockConfig(LayerConfigInterface):
    feedforward: NameAndKwargs
    sequence_mix: NameAndKwargs = None

    # will be assigned by model_config
    context_length: int = None
    num_layers: int = None
    embedding_dim: int = None
    bias: bool = None
    dropout: float = None
    norm: NormConfig = field(default_factory=lambda: NormConfig(norm_type="layer"))

    def assign_model_config_params(self, model_config: BaseTransformerConfig):
        self.context_length = model_config.context_length

        self.num_layers = model_config.num_layers
        self.embedding_dim = model_config.embedding_dim
        self.bias = model_config.bias
        self.dropout = model_config.dropout

        self.norm = model_config.norm
        self.norm.bias = self.bias

    def __post_init__(self):
        self.norm.bias = self.bias


class PreNormBlockElement(LayerInterface):
    """This is PreNormBlock wraps a layer norm followed by a layer surrounded
    by a skip connection into a block element.

    We need this to define a more fine-grained sharding strategy for FSDP.
    """

    def __init__(self, ln: LayerInterface, layer: LayerInterface):
        super().__init__()
        self.ln = ln
        self.layer = layer

    def reset_parameters(self, **kwargs):
        self.ln.reset_parameters(**kwargs)
        self.layer.reset_parameters(**kwargs)

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        return self._get_weight_decay_optim_groups_for_modules(modules=[self.ln, self.layer], **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x + self.layer(self.ln(x))


class PreNormBlock(LayerInterface):
    """A default GPT2 style block that applies a sequence_mix layer, followed by a feedforward layer with pre-norm.
    Block structure:
    * input
    - LayerNorm
    - sequence_mix
    * input + sequence_mix
    - LayerNorm
    - feedforward
    * input + feedforward
    * output
    """

    config_class = PreNormBlockConfig

    def __init__(self, config: PreNormBlockConfig, block_idx: int):
        super().__init__()
        self.block_idx = block_idx
        self.config = config
        # sequence_mix element
        ln1 = create_norm_layer(ndim=config.embedding_dim, config=config.norm)
        sequence_mix = create_sequence_mix_layer(config)
        self.seq_mix = PreNormBlockElement(ln=ln1, layer=sequence_mix)
        # ff / position-wise element
        ln2 = create_norm_layer(ndim=config.embedding_dim, config=config.norm)
        ff = create_feedforward_layer(config)
        self.ff = PreNormBlockElement(ln=ln2, layer=ff)

    def reset_parameters(self, **kwargs):
        self.seq_mix.reset_parameters(block_idx=self.block_idx, num_blocks=self.config.num_layers, **kwargs)
        self.ff.reset_parameters(block_idx=self.block_idx, num_blocks=self.config.num_layers, **kwargs)

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        return self._get_weight_decay_optim_groups_for_modules(modules=[self.seq_mix, self.ff], **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.ff(self.seq_mix(x))
        return x


@dataclass
class NPreNormBlockConfig(PreNormBlockConfig):
    sequence_mix_seq: Sequence[NameAndKwargs] = None


class NSerialPreNormBlock(LayerInterface):
    """A block that applies multiple sequence_mix layers in sequence, followed by a feedforward layer.
    Block structure:
    * input
    FOR 1..N:
        - LayerNorm
        - sequence_mix
        * input + sequence_mix
    - LayerNorm
    - feedforward
    * input + feedforward
    * output
    """

    config_class = NPreNormBlockConfig

    def __init__(self, config: NPreNormBlockConfig, block_idx: int):
        super().__init__()
        self.block_idx = block_idx
        self.config = config
        assert len(config.sequence_mix_seq) > 0, "sequence_mix_seq must be a non-empty list"

        seq_mix_seq = []
        for i, sequence_mix in enumerate(config.sequence_mix_seq):
            ln = create_norm_layer(ndim=config.embedding_dim, config=config.norm)
            seq_mix = create_sequence_mix_layer(config=config, nameandkwargs=sequence_mix)
            seq_mix_seq.append(PreNormBlockElement(ln=ln, layer=seq_mix))

        self.seq_mix_seq: nn.ModuleList = nn.ModuleList(seq_mix_seq)
        ln_ff = create_norm_layer(ndim=config.embedding_dim, config=config.norm)
        ff = create_feedforward_layer(config)
        self.ff = PreNormBlockElement(ln=ln_ff, layer=ff)

    def reset_parameters(self, **kwargs):
        for seq_mix in self.seq_mix_seq:
            seq_mix.reset_parameters(block_idx=self.block_idx, num_blocks=self.config.num_layers, **kwargs)
        self.ff.reset_parameters(block_idx=self.block_idx, num_blocks=self.config.num_layers, **kwargs)

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        return self._get_weight_decay_optim_groups_for_modules(modules=[*self.seq_mix_seq, self.ff], **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for seq_mix in self.seq_mix_seq:
            x = seq_mix(x)
        x = self.ff(x)
        return x


class NAlternatingPreNormBlock(LayerInterface):
    """A block that applies multiple sequence_mix layers followed by a feedforward layer in sequence.
    Block structure:
    * input
    FOR 1..N:
        - LayerNorm
        - sequence_mix
        * input + sequence_mix
        - LayerNorm
        - feedforward
        * input + feedforward
    * output
    """

    config_class = NPreNormBlockConfig

    def __init__(self, config: NPreNormBlockConfig, block_idx: int):
        super().__init__()
        self.block_idx = block_idx
        self.config = config

        assert len(config.sequence_mix_seq) > 0, "sequence_mix_seq must be a non-empty list"
        seq_mix_seq = []
        ff_seq = []
        for i, sequence_mix in enumerate(config.sequence_mix_seq):
            ln_seq_mix = create_norm_layer(ndim=config.embedding_dim, config=config.norm)
            seq_mix = create_sequence_mix_layer(config=config, nameandkwargs=sequence_mix)
            ln_ff = create_norm_layer(ndim=config.embedding_dim, config=config.norm)
            ff = create_feedforward_layer(config=config)
            seq_mix_seq.append(PreNormBlockElement(ln=ln_seq_mix, layer=seq_mix))
            ff_seq.append(PreNormBlockElement(ln=ln_ff, layer=ff))

        self.seq_mix_seq = nn.ModuleList(seq_mix_seq)
        self.ff_seq = nn.ModuleList(ff_seq)

    def reset_parameters(self, **kwargs):
        for seq_mix in self.seq_mix_seq:
            seq_mix.reset_parameters(block_idx=self.block_idx, num_blocks=self.config.num_layers, **kwargs)
        for ff in self.ff_seq:
            ff.reset_parameters(block_idx=self.block_idx, num_blocks=self.config.num_layers, **kwargs)

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        return self._get_weight_decay_optim_groups_for_modules(modules=[*self.seq_mix_seq, *self.ff_seq], **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for seq_mix, ff in zip(self.seq_mix_seq, self.ff_seq):
            x = ff(seq_mix(x))
        return x


_block_registry = {
    "prenorm_block": PreNormBlock,
    "nserial_pn_block": NSerialPreNormBlock,
    "nalternating_pn_block": NAlternatingPreNormBlock,
}

create_block = partial(create_layer, registry=_block_registry, layer_cfg_key="block")
