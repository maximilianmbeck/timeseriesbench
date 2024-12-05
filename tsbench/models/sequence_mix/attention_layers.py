# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..base import LayerConfigInterface, LayerInterface
from ..transformer.base import BaseTransformerConfig


@dataclass
class CausalSelfAttentionConfig(LayerConfigInterface):
    num_heads: int
    use_flash: bool = True
    attn_dropout: float = -1.0  # if negative will be same as 'dropout'
    shortname: str = ""

    # will be assigned from base model config
    embedding_dim: int = None
    dropout: float = None
    context_length: int = None
    bias: bool = None

    def assign_model_config_params(self, model_config: BaseTransformerConfig):
        self.context_length = model_config.context_length
        self.bias = model_config.bias
        self.dropout = model_config.dropout
        self.embedding_dim = model_config.embedding_dim
        if self.attn_dropout < 0.0:
            self.attn_dropout = self.dropout

    def __post_init__(self):
        if self.attn_dropout < 0.0 and self.dropout is not None:
            self.attn_dropout = self.dropout


class CausalSelfAttention(LayerInterface):
    config_class = CausalSelfAttentionConfig

    def __init__(self, config: CausalSelfAttentionConfig):
        super().__init__()
        self.config = config
        assert (
            config.embedding_dim % config.num_heads == 0
        ), f"embedding_dim ({config.embedding_dim}) must be divisible by num_heads ({config.num_heads})"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.embedding_dim, 3 * config.embedding_dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.num_heads = config.num_heads
        self.embedding_dim = config.embedding_dim
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.use_flash
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.context_length, config.context_length)).view(
                    1, 1, config.context_length, config.context_length
                ),
            )

    def reset_parameters(self, block_idx: int = 0, num_blocks: int = 1):
        def init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        self.apply(init_weights)
        if num_blocks > 0:
            # in nanogpt.py the c_proj layer is initialized with std 0.02 / math.sqrt(2 * n_layer))
            torch.nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * num_blocks))

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[set[nn.Parameter], set[nn.Parameter]]:
        # This function is overriden to make it explicit, which parameters are decayed and which are not.
        weight_decay = (self.c_attn.weight, self.c_proj.weight)
        no_weight_decay = ()
        if self.config.bias:
            no_weight_decay += (self.c_attn.bias, self.c_proj.bias)
        return weight_decay, no_weight_decay

    def forward(self, x, **kwargs):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (embedding_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.embedding_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.config.attn_dropout if self.training else 0, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MinimalAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        causal: bool = False,
        bias: bool = True,
        out_dropout: float = 0.0,
        c_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.causal = causal

        self.qkv_projection = nn.Linear(d_model, d_model * 3, bias=bias)
        self.output_projection = nn.Linear(d_model, d_model, bias=bias)
        self.out_dropout = nn.Dropout(out_dropout)
        self.c_dropout = nn.Dropout(c_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        B: batch size
        S: sequence length
        d_model: embedding dimension or model dimension
        """
        B, S, d_model = x.shape

        qkv = self.qkv_projection(x)  # (B, S, d_model * 3)
        q, k, v = qkv.split(self.d_model, dim=-1)  # (B, S, d_model)
        q = q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, S, d_head)
        k = k.view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, S, d_head)
        v = v.view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, S, d_head)

        # similarity matrix
        C_sim = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))  # (B, n_heads, S, S)
        # combination matrix
        C_comb = F.softmax(C_sim, dim=-1)  # (B, n_heads, S, S)

        if self.causal:
            # mask out the future
            C_comb = C_comb.tril(diagonal=0)  # (B, n_heads, S, S)

        C_comb = self.c_dropout(C_comb)
        y = C_comb @ v  # (B, n_heads, S, S) x (B, n_heads, S, d_head) -> (B, n_heads, S, d_head)

        y = y.transpose(1, 2).contiguous().view(B, S, d_model)  # (B, S, d_model)
        y = self.out_dropout(self.output_projection(y))
        return y
