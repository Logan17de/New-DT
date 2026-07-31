from __future__ import annotations

from typing import Iterator

import torch.nn.functional as F
from torch import Tensor, nn

from .config import DynamicTransformerConfig
from .small_hybrid_dt import SharedAttentionUniqueFFN, SmallHybridOutput
from .small_lookup_dt import LookupFFN, TokenLookupLinear
from .small_unique_attn_ffn_mod import PostActivationTokenModifier


class UniqueFFNWithPostActivationTokenMod(nn.Module):
    """Wrap an existing token-unique FFN with the projected token MOD.

    The original token-owned Up, Gate, and Down matrices are retained exactly.
    The MOD is added after the token-unique SwiGLU activation and before the
    token-unique Down projection:

        activated_t = SiLU(W_gate[t] x) * W_up[t] x
        activated'_t = activated_t + P_layer(mod_table[t])
        output_t = W_down[t] activated'_t

    ``mod_table`` is owned by the complete model and shared across all layers.
    This wrapper owns only the layer-specific projection.
    """

    def __init__(
        self,
        base_ffn: LookupFFN,
        config: DynamicTransformerConfig,
        *,
        mod_dim: int,
        mod_scale: float,
    ) -> None:
        super().__init__()
        self.base_ffn = base_ffn
        self.modifier = PostActivationTokenModifier(
            mod_dim,
            config.ffn_dim,
            init_std=config.init_std,
            scale=mod_scale,
        )

    def forward(
        self,
        x: Tensor,
        token_ids: Tensor,
        token_mod_vectors: Tensor,
    ) -> Tensor:
        up = self.base_ffn.up_proj(x, token_ids)
        gate = self.base_ffn.gate_proj(x, token_ids)
        activated = F.silu(gate) * up
        activated = activated + self.modifier(token_mod_vectors)
        return self.base_ffn.down_proj(activated, token_ids)


class SharedAttentionUniqueFFNMod(SharedAttentionUniqueFFN):
    """Shared attention + token-unique FFN + cross-layer-shared token MOD.

    Construction intentionally starts by building ``SharedAttentionUniqueFFN`` in
    full. With the same random seed, every baseline parameter is therefore exactly
    identical to the existing shared-attention/unique-FFN model. The extra token
    MOD table is zero initialized, so both models also produce exactly identical
    logits at step zero. This makes the run a paired test of the MOD contribution.
    """

    def __init__(
        self,
        config: DynamicTransformerConfig,
        *,
        mod_dim: int = 4,
        mod_scale: float = 1.0,
    ) -> None:
        super().__init__(config)
        if mod_dim <= 0:
            raise ValueError("mod_dim must be positive")
        self.mod_dim = int(mod_dim)
        self.mod_scale = float(mod_scale)
        self.architecture = (
            "shared_attn_unique_ffn_cross_layer_shared_post_activation_mod"
        )

        # Created only after the complete baseline model so the baseline parameter
        # initialization remains exactly paired under the same seed.
        self.token_mod = nn.Embedding(
            config.vocab_size,
            self.mod_dim,
            sparse=True,
        )
        nn.init.zeros_(self.token_mod.weight)

        for layer in self.layers:
            if not isinstance(layer.ffn, LookupFFN):
                raise TypeError("expected the baseline token-unique LookupFFN")
            layer.ffn = UniqueFFNWithPostActivationTokenMod(
                layer.ffn,
                config,
                mod_dim=self.mod_dim,
                mod_scale=self.mod_scale,
            )

    def sparse_parameters(self) -> Iterator[nn.Parameter]:
        yield self.embedding.weight
        yield self.token_mod.weight
        for module in self.modules():
            if isinstance(module, TokenLookupLinear):
                yield module.lookup.weight

    def dense_parameters(self) -> Iterator[nn.Parameter]:
        sparse_ids = {id(parameter) for parameter in self.sparse_parameters()}
        for parameter in self.parameters():
            if id(parameter) not in sparse_ids:
                yield parameter

    def lookup_summary(self) -> dict[str, int | float | str | bool]:
        sparse = list(self.sparse_parameters())
        dense = list(self.dense_parameters())
        unique_ffn_per_token_per_layer = (
            3 * self.config.d_model * self.config.ffn_dim
        )
        projection_per_layer = self.mod_dim * self.config.ffn_dim
        return {
            "architecture": self.architecture,
            "unique_attention": False,
            "unique_ffn": True,
            "mod_dim": self.mod_dim,
            "mod_scale": self.mod_scale,
            "ffn_mod_table_sharing": "one_token_table_shared_across_all_layers",
            "unique_ffn_parameters_per_token_per_layer": int(
                unique_ffn_per_token_per_layer
            ),
            "ffn_mod_parameters_per_token_total": self.mod_dim,
            "ffn_mod_projection_parameters_per_layer": projection_per_layer,
            "lookup_parameters": int(sum(parameter.numel() for parameter in sparse)),
            "dense_parameters": int(sum(parameter.numel() for parameter in dense)),
            "lookup_parameter_bytes": int(
                sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in sparse
                )
            ),
        }

    def forward(
        self,
        input_ids: Tensor,
        *,
        labels: Tensor | None = None,
        collect_route_grads: bool = False,
    ) -> SmallHybridOutput:
        del collect_route_grads
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        if input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size:
            raise ValueError("input_ids contain token IDs outside the vocabulary")

        x = self.embedding(input_ids)
        token_mod_vectors = self.token_mod(input_ids)
        for layer in self.layers:
            normalized = layer.attention_norm(x)
            attention_output = layer.attention(normalized)
            x = x + layer.dropout(attention_output)

            normalized = layer.ffn_norm(x)
            if not isinstance(layer.ffn, UniqueFFNWithPostActivationTokenMod):
                raise TypeError("unexpected FFN module in projected MOD model")
            ffn_output = layer.ffn(
                normalized,
                input_ids,
                token_mod_vectors,
            )
            x = x + layer.dropout(ffn_output)

        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
            )
        return SmallHybridOutput(logits=logits, loss=loss)
