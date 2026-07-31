from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch.nn.functional as F
from torch import Tensor, nn

from .config import DynamicTransformerConfig
from .layers import SharedRMSNorm
from .small_gpt import SharedLinear
from .small_lookup_dt import LookupSelfAttention, TokenLookupLinear


@dataclass(slots=True)
class UniqueAttentionFFNModOutput:
    logits: Tensor
    loss: Tensor | None = None


class PostActivationTokenModifier(nn.Module):
    """Project a shared-across-layers token MOD vector into one layer's FFN width.

    Every vocabulary token owns one small vector in the model-level MOD table.
    Every Transformer layer owns a different projection matrix. The same token
    vector is therefore reused across depth, while each layer can interpret it
    differently:

        projected_mod(token, layer) = P_layer(mod_table[token])
    """

    def __init__(
        self,
        mod_dim: int,
        out_features: int,
        *,
        init_std: float,
        scale: float,
    ) -> None:
        super().__init__()
        if mod_dim <= 0:
            raise ValueError("modifier dimension must be positive")
        self.mod_dim = int(mod_dim)
        self.out_features = int(out_features)
        self.scale = float(scale)
        self.projection = SharedLinear(
            mod_dim,
            out_features,
            init_std=init_std,
        )

    def forward(self, token_mod_vectors: Tensor) -> Tensor:
        if token_mod_vectors.shape[-1] != self.mod_dim:
            raise ValueError("token MOD vectors have the wrong final dimension")
        return self.scale * self.projection(token_mod_vectors)

    @property
    def shared_projection_parameters(self) -> int:
        return self.mod_dim * self.out_features


class SharedFFNWithPostActivationTokenMod(nn.Module):
    """Shared SwiGLU FFN with projected token MOD after activation.

    Flow:

        up = W_up x
        gate = W_gate x
        activated = SiLU(gate) * up
        activated = activated + P_layer(mod_table[token])
        output = W_down activated

    The MOD table is owned by the complete model and shared across all layers.
    This module owns only the layer-specific expansion projection.
    """

    def __init__(
        self,
        config: DynamicTransformerConfig,
        *,
        mod_dim: int,
        scale: float,
    ) -> None:
        super().__init__()
        self.up_proj = SharedLinear(
            config.d_model,
            config.ffn_dim,
            init_std=config.init_std,
        )
        self.gate_proj = SharedLinear(
            config.d_model,
            config.ffn_dim,
            init_std=config.init_std,
        )
        self.modifier = PostActivationTokenModifier(
            mod_dim,
            config.ffn_dim,
            init_std=config.init_std,
            scale=scale,
        )
        self.down_proj = SharedLinear(
            config.ffn_dim,
            config.d_model,
            init_std=config.init_std,
        )

    def forward(self, x: Tensor, token_mod_vectors: Tensor) -> Tensor:
        activated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        activated = activated + self.modifier(token_mod_vectors)
        return self.down_proj(activated)


class UniqueAttentionFFNModBlock(nn.Module):
    def __init__(
        self,
        config: DynamicTransformerConfig,
        *,
        mod_dim: int,
        mod_scale: float,
    ) -> None:
        super().__init__()
        self.attention_norm = SharedRMSNorm(config.d_model)
        self.attention = LookupSelfAttention(config)
        self.ffn_norm = SharedRMSNorm(config.d_model)
        self.ffn = SharedFFNWithPostActivationTokenMod(
            config,
            mod_dim=mod_dim,
            scale=mod_scale,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        token_ids: Tensor,
        token_mod_vectors: Tensor,
    ) -> Tensor:
        x = x + self.dropout(self.attention(self.attention_norm(x), token_ids))
        return x + self.dropout(
            self.ffn(self.ffn_norm(x), token_mod_vectors)
        )


class UniqueAttentionSharedFFNMod(nn.Module):
    """Unique attention plus shared FFN and one cross-layer token MOD table."""

    def __init__(
        self,
        config: DynamicTransformerConfig,
        *,
        mod_dim: int = 4,
        mod_scale: float = 1.0,
    ) -> None:
        super().__init__()
        config.validate()
        if mod_dim <= 0:
            raise ValueError("mod_dim must be positive")
        self.config = config
        self.mod_dim = int(mod_dim)
        self.mod_scale = float(mod_scale)

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            sparse=True,
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=config.init_std)

        # One token table is reused by every Transformer layer. This matches the
        # better-performing cross-layer-shared MOD design from the old experiments.
        self.token_mod = nn.Embedding(
            config.vocab_size,
            self.mod_dim,
            sparse=True,
        )
        nn.init.zeros_(self.token_mod.weight)

        self.layers = nn.ModuleList(
            UniqueAttentionFFNModBlock(
                config,
                mod_dim=self.mod_dim,
                mod_scale=self.mod_scale,
            )
            for _ in range(config.n_layers)
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
        yield self.token_mod.weight
        for module in self.modules():
            if isinstance(module, TokenLookupLinear):
                yield module.lookup.weight

    def dense_parameters(self) -> Iterator[nn.Parameter]:
        sparse_ids = {id(parameter) for parameter in self.sparse_parameters()}
        for parameter in self.parameters():
            if id(parameter) not in sparse_ids:
                yield parameter

    def lookup_summary(self) -> dict[str, int | float | str]:
        sparse = list(self.sparse_parameters())
        dense = list(self.dense_parameters())
        attention_per_token = 4 * self.config.d_model * self.config.d_model
        mod_projection_per_layer = self.mod_dim * self.config.ffn_dim
        return {
            "architecture": "unique_attn_shared_ffn_cross_layer_shared_post_activation_mod",
            "mod_dim": self.mod_dim,
            "mod_scale": self.mod_scale,
            "ffn_mod_table_sharing": "one_token_table_shared_across_all_layers",
            "unique_attention_parameters_per_token_per_layer": attention_per_token,
            "ffn_mod_parameters_per_token_total": self.mod_dim,
            "ffn_mod_projection_parameters_per_layer": mod_projection_per_layer,
            "lookup_parameters": int(sum(p.numel() for p in sparse)),
            "dense_parameters": int(sum(p.numel() for p in dense)),
            "lookup_parameter_bytes": int(
                sum(p.numel() * p.element_size() for p in sparse)
            ),
        }

    def forward(
        self,
        input_ids: Tensor,
        *,
        labels: Tensor | None = None,
        collect_route_grads: bool = False,
    ) -> UniqueAttentionFFNModOutput:
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
            x = layer(x, input_ids, token_mod_vectors)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
            )
        return UniqueAttentionFFNModOutput(logits=logits, loss=loss)
