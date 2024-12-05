# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from dataclasses import dataclass

import torch
from torch import nn

from ..base import LayerInterface
from ..components.act_fn import get_act_fn_cls
from ..transformer.base import BaseTransformerConfig
from .decoders import SequenceDecoder, SequenceDecoderConfig


class AanRetrievalHead(LayerInterface):
    feature_mode_input_dim_factor = {
        "singlediffmul": 4,
        "concat": 2,
    }

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        feature_mode: str = "singlediffmul",
        act_fn: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()
        assert feature_mode in list(
            self.feature_mode_input_dim_factor.keys()
        ), f"feature_mode must be one of {list(self.feature_mode_input_dim_factor.keys())}"

        self.act_fn = act_fn
        self.act_fn_cls = get_act_fn_cls(act_fn)

        self.feature_mode = feature_mode
        input_dim_factor = self.feature_mode_input_dim_factor[feature_mode]

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * input_dim_factor, embedding_dim, bias=bias),
            self.act_fn_cls(),
            nn.Linear(embedding_dim, output_dim, bias=bias),
        )

    def reset_parameters(self, **kwargs) -> None:
        for m in self.classifier:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x shape: (2*batch_size, embedding_dim)
        # the first batch_size elements are the first sequence,
        # the second batch_size elements are the second sequence
        outs = x.reshape(2, -1, x.size(-1))  # (2, batch_size, embedding_dim)
        # outs = rearrange(x, "(z b) d -> z b d", z=2)
        outs0, outs1 = outs[0], outs[1]  # (batch_size, embedding_dim)
        if self.feature_mode == "singlediffmul":
            features = torch.cat([outs0, outs1, outs0 - outs1, outs0 * outs1], dim=-1)  # (batch_size, 4*embedding_dim)
        elif self.feature_mode == "concat":
            features = torch.cat([outs0, outs1], dim=-1)
        else:
            raise ValueError(f"feature_mode {self.feature_mode} not implemented")

        logits = self.classifier(features)  # (batch_size, output_dim)
        return logits


@dataclass
class AanDecoderConfig(SequenceDecoderConfig):
    feature_mode: str = "singlediffmul"
    act_fn: str = "gelu"

    # will be assigned from base config
    bias: bool = True

    def assign_model_config_params(self, model_config: BaseTransformerConfig):
        super().assign_model_config_params(model_config)
        self.bias = model_config.bias


class AanDecoder(LayerInterface):
    config_class = AanDecoderConfig

    def __init__(self, config: AanDecoderConfig):
        super().__init__()
        self.config = config
        assert len(self.config.output_dim) == 1, "AanDecoder only supports single output_dim"
        output_dim = config.output_dim[0]
        config.output_dim = None  # set this to None so that SequenceDecoder has the Identity() as output_transform
        self.sequence_decoder = SequenceDecoder(config)

        self.retrieval_head = AanRetrievalHead(
            embedding_dim=config.embedding_dim,
            output_dim=output_dim,
            feature_mode=config.feature_mode,
            act_fn=config.act_fn,
            bias=config.bias,
        )

    def reset_parameters(self, **kwargs) -> None:
        self.sequence_decoder.reset_parameters(**kwargs)
        self.retrieval_head.reset_parameters(**kwargs)

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[set[nn.Parameter], set[nn.Parameter]]:
        return self._get_weight_decay_optim_groups_for_modules(
            modules=[self.sequence_decoder, self.retrieval_head], **kwargs
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.sequence_decoder(x, **kwargs)
        x = self.retrieval_head(x, **kwargs)
        return x
