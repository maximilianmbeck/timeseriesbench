# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from torch import nn

from ..base import LayerInterface


class CausalDepthwiseConv1d(LayerInterface):
    """
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,F)

    Args:
        feature_dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        causal_conv_bias: whether to use bias in the depthwise convolution
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                        If True, it mixes the convolved features across channels.
                        If False, all the features are convolved independently.
    """

    def __init__(self, feature_dim, kernel_size=3, causal_conv_bias=False, channel_mixing=False, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim  # F
        self.kernel_size = kernel_size
        self.causal_conv_bias = causal_conv_bias
        self.groups = self.feature_dim
        if channel_mixing:
            self.groups = 1
        self.pad = self.kernel_size - 1  # padding of this size assures temporal causality.
        self.conv = nn.Conv1d(
            self.feature_dim,
            self.feature_dim,
            self.kernel_size,
            padding=self.pad,
            groups=self.groups,
            bias=self.causal_conv_bias,
            **kwargs
        )
        # B, C, L
        self.reset_parameters()

    def reset_parameters(self, **kwargs):
        self.conv.reset_parameters()

    def _create_weight_decay_optim_groups(self) -> tuple[set[nn.Parameter], set[nn.Parameter]]:
        weight_decay = (self.conv.weight,)
        no_weight_decay = ()
        if self.causal_conv_bias:
            no_weight_decay += (self.conv.bias,)
        return weight_decay, no_weight_decay

    def forward(self, x):
        y = x.transpose(2, 1)  # (B,F,T) tensor - now in the right shape for conv layer.
        y = self.conv(y)  # (B,F,T+pad) tensor
        return y[:, :, : -self.pad].transpose(2, 1)
