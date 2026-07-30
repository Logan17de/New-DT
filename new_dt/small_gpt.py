from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import DynamicTransformerConfig
from .layers import RotaryEmbedding, SharedRMSNorm


@dataclass(slots=True)
class SmallGPTOutput:
    logits: Tensor
    loss: Tensor | None = None


class SharedLinear(nn.Linear):
    """Bias-free conventional projection with the same marginal init as sDT."""

    def __init__(self, in_features: int, out_features: int, *, init_std: float) -> None:
        super().__init__(in_features, out_features, bias=False)
        nn.init.normal_(
            self.weight,
            mean=0.0,
            std=init_std / math.sqrt(max(in_features, 1)),
        )


class SharedSelfAttention(nn.Module):
    def __init__(self, config: DynamicTransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.rope = RotaryEmbedding(self.head_dim, theta=config.rope_theta)
        self.q_proj = SharedLinear(
            config.d_model, config.d_model, init_std=config.init_std
        )
        self.k_proj = SharedLinear(
            config.d_model, config.d_model, init_std=config.init_std
        )
        self.v_proj = SharedLinear(
            config.d_model, config.d_model, init_std=config.init_std
        )
        self.o_proj = SharedLinear(
            config.d_model, config.d_model, init_std=config.init_std
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, sequence, _ = x.shape
        return x.view(batch, sequence, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        q, k = self.rope(q, k)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        batch, _, sequence, _ = attended.shape
        merged = attended.transpose(1, 2).contiguous().view(batch, sequence, -1)
        return self.o_proj(merged)


class SharedFFN(nn.Module):
    def __init__(self, config: DynamicTransformerConfig) -> None:
        super().__init__()
        self.up_proj = SharedLinear(
            config.d_model, config.ffn_dim, init_std=config.init_std
        )
        self.gate_proj = SharedLinear(
            config.d_model, config.ffn_dim, init_std=config.init_std
        )
        self.down_proj = SharedLinear(
            config.ffn_dim, config.d_model, init_std=config.init_std
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SharedTransformerBlock(nn.Module):
    def __init__(self, config: DynamicTransformerConfig) -> None:
        super().__init__()
        self.attention_norm = SharedRMSNorm(config.d_model)
        self.attention = SharedSelfAttention(config)
        self.ffn_norm = SharedRMSNorm(config.d_model)
        self.ffn = SharedFFN(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attention(self.attention_norm(x)))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class SmallGPT(nn.Module):
    """Conventional shared-matrix GPT baseline matched to the sDT topology.

    Both models use RMSNorm, RoPE, SwiGLU, causal SDPA, the same depth/width, and
    separate input and output weights. The difference is that this model uses one
    conventional Q/K/V/O and FFN matrix per layer, while sDT reconstructs those
    matrices from token-owned scalar routes.
    """

    def __init__(self, config: DynamicTransformerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=config.init_std)
        self.layers = nn.ModuleList(
            SharedTransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = SharedRMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=config.init_std)

    @property
    def lm_is_untied(self) -> bool:
        return (
            self.embedding.weight is not self.lm_head.weight
            and self.embedding.weight.data_ptr() != self.lm_head.weight.data_ptr()
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        labels: Tensor | None = None,
        collect_route_grads: bool = False,
    ) -> SmallGPTOutput:
        del collect_route_grads  # Accepted so both models share one training API.
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        if input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size:
            raise ValueError("input_ids contain token IDs outside the vocabulary")

        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
            )
        return SmallGPTOutput(logits=logits, loss=loss)
