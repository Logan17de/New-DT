from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import DynamicTransformerConfig
from .pools import RoutedParameterTensor


def _routing_kwargs(config: DynamicTransformerConfig) -> dict[str, int | float]:
    return {
        "page_size": config.route_page_size,
        "templates_per_page": config.route_templates_per_page,
        "delta_promotion_threshold": config.route_delta_promotion_threshold,
        "template_promotion_threshold": config.route_template_promotion_threshold,
        "template_promotion_fraction": config.route_template_promotion_fraction,
        "shared_delta_min_reuse": config.route_shared_delta_min_reuse,
        "cache_pages": config.route_cache_pages,
        "selector_dense_promotion_fraction": config.route_selector_dense_promotion_fraction,
        "selector_dense_demotion_fraction": config.route_selector_dense_demotion_fraction,
        "materialize_token_chunk": config.route_materialize_token_chunk,
    }


class SharedRMSNorm(nn.Module):
    """Shared numerical stabilization; it is not token-owned capacity."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """RoPE rotations with a device/dtype-aware cosine/sine cache."""

    def __init__(self, head_dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self._rope_cache: dict[tuple[str, torch.dtype, int], tuple[Tensor, Tensor]] = {}

    @staticmethod
    def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def _cos_sin(self, seq_len: int, reference: Tensor) -> tuple[Tensor, Tensor]:
        key = (str(reference.device), reference.dtype, seq_len)
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        positions = torch.arange(seq_len, device=reference.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq.to(device=reference.device))
        cos = angles.cos().to(dtype=reference.dtype).view(1, 1, seq_len, -1)
        sin = angles.sin().to(dtype=reference.dtype).view(1, 1, seq_len, -1)
        self._rope_cache[key] = (cos, sin)
        return cos, sin

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        cos, sin = self._cos_sin(q.shape[-2], q)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)


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
            **_routing_kwargs(config),
        )

    def forward(self, token_ids: Tensor, *, collect_route_grads: bool = False) -> Tensor:
        return self.parameters_by_token(token_ids, collect_route_grads=collect_route_grads)


class TokenRoutedLinear(nn.Module):
    """Token-specific matrix multiplication with bounded route materialization."""

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
        self.out_tile = min(config.route_linear_out_tile, out_features)
        self.parameters_by_token = RoutedParameterTensor(
            config.vocab_size,
            (out_features, in_features),
            shared_fraction=config.initial_shared_fraction,
            growth_factor=config.pool_growth_factor,
            init_std=config.init_std / math.sqrt(max(in_features, 1)),
            name=name,
            **_routing_kwargs(config),
        )

    def forward(
        self,
        x: Tensor,
        token_ids: Tensor,
        *,
        collect_route_grads: bool = False,
    ) -> Tensor:
        outputs: list[Tensor] = []
        for out_start in range(0, self.out_features, self.out_tile):
            out_stop = min(self.out_features, out_start + self.out_tile)
            route_start = out_start * self.in_features
            route_stop = out_stop * self.in_features
            flat_weights = self.parameters_by_token.forward_slice(
                token_ids,
                route_start,
                route_stop,
                collect_route_grads=collect_route_grads,
            )
            weights = flat_weights.view(
                *token_ids.shape, out_stop - out_start, self.in_features
            )
            outputs.append(torch.einsum("...oi,...i->...o", weights, x))
        return torch.cat(outputs, dim=-1) if len(outputs) > 1 else outputs[0]


class DynamicSelfAttention(nn.Module):
    def __init__(self, config: DynamicTransformerConfig, layer_index: int) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.rope = RotaryEmbedding(self.head_dim, theta=config.rope_theta)
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
        q, k = self.rope(q, k)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        batch, _, seq, _ = attended.shape
        attended = attended.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.o_proj(attended, token_ids, collect_route_grads=collect_route_grads)


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
        return self.down_proj(hidden, token_ids, collect_route_grads=collect_route_grads)


class DynamicTransformerBlock(nn.Module):
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
    """Candidate vectors reconstructed in vocabulary tiles to cap peak memory."""

    def __init__(self, config: DynamicTransformerConfig) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
        self.vocab_tile = min(config.route_lm_head_tile, config.vocab_size)
        self.parameters_by_token = RoutedParameterTensor(
            config.vocab_size,
            (config.d_model,),
            shared_fraction=config.initial_shared_fraction,
            growth_factor=config.pool_growth_factor,
            init_std=config.init_std,
            name="lm_head",
            **_routing_kwargs(config),
        )

    def forward(self, x: Tensor, *, collect_route_grads: bool = False) -> Tensor:
        logits: list[Tensor] = []
        for start in range(0, self.vocab_size, self.vocab_tile):
            stop = min(self.vocab_size, start + self.vocab_tile)
            candidate_ids = torch.arange(start, stop, device=x.device)
            output_vectors = self.parameters_by_token(
                candidate_ids, collect_route_grads=collect_route_grads
            )
            logits.append(torch.einsum("btd,vd->btv", x, output_vectors))
        return torch.cat(logits, dim=-1) if len(logits) > 1 else logits[0]
