from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import DynamicTransformerConfig
from .pools import RoutedParameterTensor


class SharedRMSNorm(nn.Module):
    """Shared numerical stabilization; it is not token-owned capacity."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class TokenRoutedEmbedding(nn.Module):
    def __init__(self, config: DynamicTransformerConfig) -> None:
        super().__init__()
        self.parameters_by_token = RoutedParameterTensor(
            config.vocab_size,
            (config.d_model,),
            shared_fraction=config.initial_shared_fraction,
            growth_factor=config.pool_growth_factor,
            init_std=config.init_std,
            name="embedding",
        )

    def forward(self, token_ids: Tensor, *, collect_route_grads: bool = False) -> Tensor:
        return self.parameters_by_token(token_ids, collect_route_grads=collect_route_grads)


class TokenRoutedLinear(nn.Module):
    """A full token-specific matrix assembled from one scalar pool."""

    def __init__(
        self,
        config: DynamicTransformerConfig,
        in_features: int,
        out_features: int,
        *,
        name: str,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.parameters_by_token = RoutedParameterTensor(
            config.vocab_size,
            (out_features, in_features),
            shared_fraction=config.initial_shared_fraction,
            growth_factor=config.pool_growth_factor,
            init_std=config.init_std / math.sqrt(max(in_features, 1)),
            name=name,
        )

    def forward(
        self,
        x: Tensor,
        token_ids: Tensor,
        *,
        collect_route_grads: bool = False,
    ) -> Tensor:
        weights = self.parameters_by_token(
            token_ids, collect_route_grads=collect_route_grads
        )
        return torch.einsum("...oi,...i->...o", weights, x)


class DynamicSelfAttention(nn.Module):
    def __init__(self, config: DynamicTransformerConfig, layer_index: int) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        prefix = f"attention.layer_{layer_index}"
        self.q_proj = TokenRoutedLinear(
            config, config.d_model, config.d_model, name=f"{prefix}.q"
        )
        self.k_proj = TokenRoutedLinear(
            config, config.d_model, config.d_model, name=f"{prefix}.k"
        )
        self.v_proj = TokenRoutedLinear(
            config, config.d_model, config.d_model, name=f"{prefix}.v"
        )
        self.o_proj = TokenRoutedLinear(
            config, config.d_model, config.d_model, name=f"{prefix}.o"
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        token_ids: Tensor,
        *,
        collect_route_grads: bool = False,
    ) -> Tensor:
        q = self._split_heads(
            self.q_proj(x, token_ids, collect_route_grads=collect_route_grads)
        )
        k = self._split_heads(
            self.k_proj(x, token_ids, collect_route_grads=collect_route_grads)
        )
        v = self._split_heads(
            self.v_proj(x, token_ids, collect_route_grads=collect_route_grads)
        )
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        batch, _, seq, _ = attended.shape
        attended = attended.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.o_proj(
            attended, token_ids, collect_route_grads=collect_route_grads
        )


class DynamicFFN(nn.Module):
    def __init__(self, config: DynamicTransformerConfig, layer_index: int) -> None:
        super().__init__()
        prefix = f"ffn.layer_{layer_index}"
        self.up_proj = TokenRoutedLinear(
            config, config.d_model, config.ffn_dim, name=f"{prefix}.up"
        )
        self.gate_proj = TokenRoutedLinear(
            config, config.d_model, config.ffn_dim, name=f"{prefix}.gate"
        )
        self.down_proj = TokenRoutedLinear(
            config, config.ffn_dim, config.d_model, name=f"{prefix}.down"
        )

    def forward(
        self,
        x: Tensor,
        token_ids: Tensor,
        *,
        collect_route_grads: bool = False,
    ) -> Tensor:
        up = self.up_proj(x, token_ids, collect_route_grads=collect_route_grads)
        gate = self.gate_proj(x, token_ids, collect_route_grads=collect_route_grads)
        hidden = F.silu(gate) * up
        return self.down_proj(
            hidden, token_ids, collect_route_grads=collect_route_grads
        )


class DynamicTransformerBlock(nn.Module):
    """Token-owned attention/FFN surrounded by shared norm and residual plumbing."""

    def __init__(self, config: DynamicTransformerConfig, layer_index: int) -> None:
        super().__init__()
        self.attention_norm = SharedRMSNorm(config.d_model)
        self.attention = DynamicSelfAttention(config, layer_index)
        self.ffn_norm = SharedRMSNorm(config.d_model)
        self.ffn = DynamicFFN(config, layer_index)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        token_ids: Tensor,
        *,
        collect_route_grads: bool = False,
    ) -> Tensor:
        # Residual addition is common and parameter-free.
        x = x + self.dropout(
            self.attention(
                self.attention_norm(x),
                token_ids,
                collect_route_grads=collect_route_grads,
            )
        )
        x = x + self.dropout(
            self.ffn(
                self.ffn_norm(x),
                token_ids,
                collect_route_grads=collect_route_grads,
            )
        )
        return x


class TokenRoutedLMHead(nn.Module):
    """Candidate output tokens own their scalar-routed LM vectors."""

    def __init__(self, config: DynamicTransformerConfig) -> None:
        super().__init__()
        self.parameters_by_token = RoutedParameterTensor(
            config.vocab_size,
            (config.d_model,),
            shared_fraction=config.initial_shared_fraction,
            growth_factor=config.pool_growth_factor,
            init_std=config.init_std,
            name="lm_head",
        )

    def forward(self, x: Tensor, *, collect_route_grads: bool = False) -> Tensor:
        output_vectors = self.parameters_by_token.materialize_all(
            collect_route_grads=collect_route_grads
        )
        return torch.einsum("btd,vd->btv", x, output_vectors)
