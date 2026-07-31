from __future__ import annotations

from typing import Iterator

import torch.nn.functional as F
from torch import Tensor, nn

from .config import DynamicTransformerConfig
from .small_gpt import SharedLinear, SharedSelfAttention
from .small_hybrid_dt import SharedAttentionUniqueFFN, SmallHybridOutput
from .small_lookup_dt import LookupFFN, TokenLookupLinear


class AttentionTokenModifier(nn.Module):
    """Project one token-specific MOD vector into the attention output width.

    The complete model owns one sparse token MOD table shared across depth. Each
    Transformer layer owns one independent dense projection:

        attention_mod(token, layer) = scale * P_layer(mod_table[token])

    The projected vector is added to the shared attention output before the
    attention residual connection is completed.
    """

    def __init__(
        self,
        mod_dim: int,
        d_model: int,
        *,
        init_std: float,
        scale: float,
    ) -> None:
        super().__init__()
        if mod_dim <= 0:
            raise ValueError("modifier dimension must be positive")
        self.mod_dim = int(mod_dim)
        self.d_model = int(d_model)
        self.scale = float(scale)
        self.projection = SharedLinear(
            self.mod_dim,
            self.d_model,
            init_std=init_std,
        )

    def forward(self, token_mod_vectors: Tensor) -> Tensor:
        if token_mod_vectors.shape[-1] != self.mod_dim:
            raise ValueError("token MOD vectors have the wrong final dimension")
        return self.scale * self.projection(token_mod_vectors)


class SharedAttentionModUniqueFFN(SharedAttentionUniqueFFN):
    """Shared attention + token-specific Attention MOD + token-unique FFN.

    Construction starts by building the complete existing
    ``SharedAttentionUniqueFFN`` baseline. The extra MOD table is zero initialized,
    so with the same seed every baseline parameter, step-zero logit, and step-zero
    loss is exactly identical to that baseline.

    For token ``t`` in layer ``l``:

        A[t,l]  = SharedAttention_l(Norm(h[t,l]))
        D[t,l]  = scale * P_l(mod_table[t])
        h'[t,l] = h[t,l] + Dropout(A[t,l] + D[t,l])
        y[t,l]  = h'[t,l] + Dropout(UniqueFFN_l(Norm(h'[t,l]), t))

    One token MOD vector is reused across every layer, while every layer has a
    distinct projection from MOD width to ``d_model``.
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
            "shared_attn_cross_layer_shared_token_attn_mod_unique_ffn"
        )

        # These modules are deliberately created only after the complete paired
        # baseline has been initialized.
        self.token_mod = nn.Embedding(
            config.vocab_size,
            self.mod_dim,
            sparse=True,
        )
        nn.init.zeros_(self.token_mod.weight)

        for layer in self.layers:
            if not isinstance(layer.attention, SharedSelfAttention):
                raise TypeError("expected the paired shared-attention baseline")
            if not isinstance(layer.ffn, LookupFFN):
                raise TypeError("expected the paired token-unique FFN baseline")
            layer.attention_modifier = AttentionTokenModifier(
                self.mod_dim,
                config.d_model,
                init_std=config.init_std,
                scale=self.mod_scale,
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
        projection_per_layer = self.mod_dim * self.config.d_model
        return {
            "architecture": self.architecture,
            "unique_attention": False,
            "attention_mod": True,
            "unique_ffn": True,
            "mod_dim": self.mod_dim,
            "mod_scale": self.mod_scale,
            "attention_mod_table_sharing": (
                "one_token_table_shared_across_all_layers"
            ),
            "attention_mod_placement": (
                "after_shared_attention_output_before_attention_residual"
            ),
            "unique_ffn_parameters_per_token_per_layer": int(
                unique_ffn_per_token_per_layer
            ),
            "attention_mod_parameters_per_token_total": self.mod_dim,
            "attention_mod_projection_parameters_per_layer": projection_per_layer,
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
            modifier = getattr(layer, "attention_modifier", None)
            if not isinstance(modifier, AttentionTokenModifier):
                raise TypeError("missing projected Attention MOD module")
            attention_output = attention_output + modifier(token_mod_vectors)
            x = x + layer.dropout(attention_output)

            normalized = layer.ffn_norm(x)
            if not isinstance(layer.ffn, LookupFFN):
                raise TypeError("unexpected FFN module in Attention MOD model")
            ffn_output = layer.ffn(normalized, input_ids)
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
