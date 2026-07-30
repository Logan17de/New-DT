from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import DynamicTransformerConfig
from .layers import RotaryEmbedding, SharedRMSNorm


@dataclass(slots=True)
class SmallLookupDTOutput:
    logits: Tensor
    loss: Tensor | None = None


class TokenLookupLinear(nn.Module):
    """One complete bias-free matrix per vocabulary token.

    The lookup weight has shape ``[vocab, out_features * in_features]``. Only rows
    selected by the current input token IDs participate in a forward pass. Sparse
    embedding gradients therefore update only token tables that were actually used.
    """

    def __init__(
        self,
        vocab_size: int,
        in_features: int,
        out_features: int,
        *,
        init_std: float,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.in_features = in_features
        self.out_features = out_features
        self.lookup = nn.Embedding(
            vocab_size,
            out_features * in_features,
            sparse=True,
        )
        nn.init.normal_(
            self.lookup.weight,
            mean=0.0,
            std=init_std / math.sqrt(max(in_features, 1)),
        )

    def forward(self, x: Tensor, token_ids: Tensor) -> Tensor:
        if x.shape[:-1] != token_ids.shape:
            raise ValueError("x and token_ids must have matching leading dimensions")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input width {self.in_features}, received {x.shape[-1]}"
            )
        flat = self.lookup(token_ids)
        weights = flat.view(*token_ids.shape, self.out_features, self.in_features)
        return torch.einsum("...oi,...i->...o", weights, x)

    @property
    def parameter_count(self) -> int:
        return self.vocab_size * self.out_features * self.in_features


class LookupSelfAttention(nn.Module):
    def __init__(self, config: DynamicTransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.rope = RotaryEmbedding(self.head_dim, theta=config.rope_theta)
        kwargs = {
            "vocab_size": config.vocab_size,
            "in_features": config.d_model,
            "out_features": config.d_model,
            "init_std": config.init_std,
        }
        self.q_proj = TokenLookupLinear(**kwargs)
        self.k_proj = TokenLookupLinear(**kwargs)
        self.v_proj = TokenLookupLinear(**kwargs)
        self.o_proj = TokenLookupLinear(**kwargs)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, sequence, _ = x.shape
        return x.view(batch, sequence, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor, token_ids: Tensor) -> Tensor:
        q = self._split_heads(self.q_proj(x, token_ids))
        k = self._split_heads(self.k_proj(x, token_ids))
        v = self._split_heads(self.v_proj(x, token_ids))
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
        return self.o_proj(merged, token_ids)


class LookupFFN(nn.Module):
    def __init__(self, config: DynamicTransformerConfig) -> None:
        super().__init__()
        self.up_proj = TokenLookupLinear(
            config.vocab_size,
            config.d_model,
            config.ffn_dim,
            init_std=config.init_std,
        )
        self.gate_proj = TokenLookupLinear(
            config.vocab_size,
            config.d_model,
            config.ffn_dim,
            init_std=config.init_std,
        )
        self.down_proj = TokenLookupLinear(
            config.vocab_size,
            config.ffn_dim,
            config.d_model,
            init_std=config.init_std,
        )

    def forward(self, x: Tensor, token_ids: Tensor) -> Tensor:
        up = self.up_proj(x, token_ids)
        gate = self.gate_proj(x, token_ids)
        return self.down_proj(F.silu(gate) * up, token_ids)


class LookupTransformerBlock(nn.Module):
    def __init__(self, config: DynamicTransformerConfig) -> None:
        super().__init__()
        self.attention_norm = SharedRMSNorm(config.d_model)
        self.attention = LookupSelfAttention(config)
        self.ffn_norm = SharedRMSNorm(config.d_model)
        self.ffn = LookupFFN(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, token_ids: Tensor) -> Tensor:
        x = x + self.dropout(self.attention(self.attention_norm(x), token_ids))
        return x + self.dropout(self.ffn(self.ffn_norm(x), token_ids))


class SmallLookupDT(nn.Module):
    """Small DT ablation with explicit token-owned lookup tables.

    There is no scalar pool, route sharing, SPRC, split, merge, compaction, or
    structural controller. Every vocabulary token owns an independent embedding row
    and independent Q/K/V/O and SwiGLU matrices in every layer. The untied LM output
    table has one independent candidate vector per output token, as in a standard
    untied language-model head.
    """

    def __init__(self, config: DynamicTransformerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            sparse=True,
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=config.init_std)
        self.layers = nn.ModuleList(
            LookupTransformerBlock(config) for _ in range(config.n_layers)
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

    def sparse_parameters(self) -> Iterator[nn.Parameter]:
        yield self.embedding.weight
        for module in self.modules():
            if isinstance(module, TokenLookupLinear):
                yield module.lookup.weight

    def dense_parameters(self) -> Iterator[nn.Parameter]:
        sparse_ids = {id(parameter) for parameter in self.sparse_parameters()}
        for parameter in self.parameters():
            if id(parameter) not in sparse_ids:
                yield parameter

    def lookup_summary(self) -> dict[str, int]:
        sparse = sum(parameter.numel() for parameter in self.sparse_parameters())
        dense = sum(parameter.numel() for parameter in self.dense_parameters())
        per_token_layer = (
            4 * self.config.d_model * self.config.d_model
            + 3 * self.config.d_model * self.config.ffn_dim
        )
        return {
            "lookup_parameters": int(sparse),
            "dense_parameters": int(dense),
            "token_owned_parameters_per_layer_per_token": int(per_token_layer),
            "lookup_parameter_bytes": int(
                sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in self.sparse_parameters()
                )
            ),
        }

    def forward(
        self,
        input_ids: Tensor,
        *,
        labels: Tensor | None = None,
        collect_route_grads: bool = False,
    ) -> SmallLookupDTOutput:
        del collect_route_grads
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        if input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size:
            raise ValueError("input_ids contain token IDs outside the vocabulary")

        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, input_ids)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
            )
        return SmallLookupDTOutput(logits=logits, loss=loss)
