# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn

from ..base import LayerConfigInterface, LayerInterface
from ..components.identity import Identity
from ..components.linear import Linear, LinearConfig
from ..transformer.base import BaseTransformerConfig

# In the safari / hyena repo the sequence decoder has different modes:
# last, first, pool. These modes describe which elements from the sequence should
# be used to generate the output.
# last: use the last len_output elements from the sequence
# first: use the first len_output elements from the sequence
# pool: use the mean of the sequence

# They also have an option `l_output` (len_output) which specifies how many elements
# of the sequence should be considered for the output.
# This is only used for the last and first mode.

"""In this module we implement different 'decoders'.
A decoder is the last part of a sequence model and decodes the last embedding sequence, depending on the
specified task."""


@dataclass
class SequenceDecoderConfig(LayerConfigInterface):
    len_output: int = 1
    # last: use the last len_output elements from the sequence
    # first: use the first len_output elements from the sequence
    # pool: use the mean of the sequence
    agg_mode: str = "pool"  # options: last, first, pool
    use_lengths: bool = True  # whether to use the lengths

    # will be assigned from base model config
    bias: bool = True
    output_dim: Sequence[int] = None
    embedding_dim: int = None
    context_length: int = None

    def assign_model_config_params(self, model_config: BaseTransformerConfig):
        self.context_length = model_config.context_length
        self.embedding_dim = model_config.embedding_dim
        self.output_dim = model_config.output_dim
        self.bias = model_config.bias

    def __post_init__(self):
        assert self.agg_mode in ["last", "first", "pool"], f"agg_mode must be one of ['last', 'first', 'pool']"
        if self.len_output != 1:
            assert self.agg_mode != "pool", "agg_mode 'pool' is not supported for len_output > 1"


class SequenceDecoder(LayerInterface):
    config_class = SequenceDecoderConfig

    def __init__(self, config: SequenceDecoderConfig):
        super().__init__()
        assert config.output_dim is None or len(config.output_dim) in [
            1,
            2,
        ], "output_dim must be a tuple of length 1 or 2"
        if len(config.output_dim) == 2:
            assert config.output_dim[1] == 1, f"Second dimension of output_dim must be 1. Got: {config.output_dim[1]}"
        self.config = config

        self.sequence_agg_fn = self._get_seq_agg_fn()

        self.output_transform = (
            Identity()
            if config.output_dim is None
            else Linear(
                config=LinearConfig(
                    in_features=config.embedding_dim, out_features=config.output_dim[0], bias=self.config.bias
                )
            )
        )

    def reset_parameters(self, **kwargs):
        self.output_transform.reset_parameters()

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[set[nn.Parameter], set[nn.Parameter]]:
        return self.output_transform.get_weight_decay_optim_groups(**kwargs)

    def _get_seq_agg_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self.config.agg_mode == "last":
            return lambda x: x[..., -self.config.len_output :, :]
        elif self.config.agg_mode == "first":
            return lambda x: x[..., : self.config.len_output, :]
        elif self.config.agg_mode == "pool":
            return lambda x: x.mean(dim=-2, keepdim=True)
        else:
            raise NotImplementedError(f"agg_mode {self.config.agg_mode} not implemented")

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None, **kwargs) -> torch.Tensor:
        # batch of sequences x shape: (batch_size, context_length, embedding_dim)

        # extract only the elements within [0:length] of each sequence in the batch
        # see https://github.com/HazyResearch/safari/blob/26f6223254c241ec3418a0360a3a704e0a24d73d/src/tasks/decoders.py#L120
        if self.config.use_lengths:
            assert lengths is not None, "lengths must be specified if use_lengths is True"
            # TODO make this more efficient by avoiding the for loop over the batch dimension.
            # Figure out if there are pytorch built-in function for this, i.e. PackedSequence or so.
            x = torch.stack(
                [
                    self.sequence_agg_fn(single_sequence[..., :length, :])
                    for single_sequence, length in zip(x.unbind(dim=0), lengths)
                ],
                dim=0,
            )
        else:
            x = self.sequence_agg_fn(x)

        # shape: (batch_size, len_output, embedding_dim)
        if x.size(-2) == 1:
            x = x.squeeze(dim=-2)
        x = self.output_transform(x)
        return x
