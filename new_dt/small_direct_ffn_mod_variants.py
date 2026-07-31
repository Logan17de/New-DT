from __future__ import annotations

from typing import Iterator, Literal

import torch.nn.functional as F
from torch import Tensor, nn

from .config import DynamicTransformerConfig
from .small_hybrid_dt import SharedAttentionUniqueFFN, SmallHybridOutput
from .small_lookup_dt import LookupFFN, TokenLookupLinear


DirectFFNModPlacement = Literal[
    "branch_pre_activation",
    "gate_pre_activation",
    "post_activation",
]


class DirectTokenModBroadcast(nn.Module):
    """Broadcast a small token MOD vector to FFN width without learned projection.

    For ``mod_dim=4`` and ``ffn_dim=64``, the four token-owned values are tiled
    sixteen times:

        [m0, m1, m2, m3] -> [m0, m1, m2, m3, ..., m0, m1, m2, m3]

    This is a fixed reshape/broadcast operation. It has no parameters and no
    learned projection matrix.
    """

    def __init__(self, mod_dim: int, ffn_dim: int, *, scale: float) -> None:
        super().__init__()
        if mod_dim <= 0:
            raise ValueError("modifier dimension must be positive")
        if ffn_dim <= 0:
            raise ValueError("FFN dimension must be positive")
        if ffn_dim % mod_dim != 0:
            raise ValueError(
                "direct MOD requires ffn_dim to be divisible by mod_dim: "
                f"ffn_dim={ffn_dim}, mod_dim={mod_dim}"
            )
        self.mod_dim = int(mod_dim)
        self.ffn_dim = int(ffn_dim)
        self.repeats = self.ffn_dim // self.mod_dim
        self.scale = float(scale)

    def forward(self, token_mod_vectors: Tensor) -> Tensor:
        if token_mod_vectors.shape[-1] != self.mod_dim:
            raise ValueError("token MOD vectors have the wrong final dimension")
        repeat_factors = (1,) * (token_mod_vectors.ndim - 1) + (self.repeats,)
        expanded = token_mod_vectors.repeat(*repeat_factors)
        if expanded.shape[-1] != self.ffn_dim:
            raise RuntimeError("direct MOD broadcast produced the wrong FFN width")
        return self.scale * expanded


class UniqueFFNWithDirectTokenMod(nn.Module):
    """Token-unique SwiGLU FFN with one direct additive MOD placement."""

    def __init__(
        self,
        base_ffn: LookupFFN,
        config: DynamicTransformerConfig,
        *,
        mod_dim: int,
        mod_scale: float,
        placement: DirectFFNModPlacement,
    ) -> None:
        super().__init__()
        if placement not in (
            "branch_pre_activation",
            "gate_pre_activation",
            "post_activation",
        ):
            raise ValueError(f"unsupported direct MOD placement: {placement}")
        self.base_ffn = base_ffn
        self.placement = placement
        self.broadcast = DirectTokenModBroadcast(
            mod_dim,
            config.ffn_dim,
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
        direct_mod = self.broadcast(token_mod_vectors)

        if self.placement == "branch_pre_activation":
            activated = F.silu(gate) * (up + direct_mod)
        elif self.placement == "gate_pre_activation":
            activated = F.silu(gate + direct_mod) * up
        else:
            activated = F.silu(gate) * up + direct_mod

        return self.base_ffn.down_proj(activated, token_ids)


class SharedAttentionUniqueFFNDirectMod(SharedAttentionUniqueFFN):
    """Paired Shared-ATTN + unique-FFN model with direct token MOD.

    The complete baseline is initialized first. A single zero-initialized token
    MOD table is then added and reused by every Transformer layer. Since there is
    no projection matrix and the MOD begins at zero, all three placement variants
    exactly match the baseline at step zero under the same seed.
    """

    def __init__(
        self,
        config: DynamicTransformerConfig,
        *,
        placement: DirectFFNModPlacement,
        mod_dim: int = 4,
        mod_scale: float = 1.0,
    ) -> None:
        super().__init__(config)
        if mod_dim <= 0:
            raise ValueError("mod_dim must be positive")
        if config.ffn_dim % mod_dim != 0:
            raise ValueError(
                "direct MOD requires ffn_dim divisible by mod_dim"
            )
        self.placement = placement
        self.mod_dim = int(mod_dim)
        self.mod_scale = float(mod_scale)
        self.architecture = f"shared_attn_unique_ffn_direct_mod_{placement}"

        # Created after the complete paired baseline to preserve all baseline
        # initialization exactly. One table is shared across every layer.
        self.token_mod = nn.Embedding(
            config.vocab_size,
            self.mod_dim,
            sparse=True,
        )
        nn.init.zeros_(self.token_mod.weight)

        for layer in self.layers:
            if not isinstance(layer.ffn, LookupFFN):
                raise TypeError("expected the baseline token-unique LookupFFN")
            layer.ffn = UniqueFFNWithDirectTokenMod(
                layer.ffn,
                config,
                mod_dim=self.mod_dim,
                mod_scale=self.mod_scale,
                placement=self.placement,
            )

    @property
    def direct_mod_has_projection(self) -> bool:
        return False

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
        return {
            "architecture": self.architecture,
            "unique_attention": False,
            "unique_ffn": True,
            "direct_mod_placement": self.placement,
            "mod_dim": self.mod_dim,
            "mod_scale": self.mod_scale,
            "mod_projection": False,
            "mod_table_sharing": "one_token_table_shared_across_all_layers",
            "mod_broadcast": "fixed_periodic_tile_to_ffn_width",
            "mod_parameters_per_token_total": self.mod_dim,
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
            attention_output = layer.attention(layer.attention_norm(x))
            x = x + layer.dropout(attention_output)

            if not isinstance(layer.ffn, UniqueFFNWithDirectTokenMod):
                raise TypeError("unexpected FFN module in direct MOD model")
            ffn_output = layer.ffn(
                layer.ffn_norm(x),
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


class BranchPreActivationDirectMod(SharedAttentionUniqueFFNDirectMod):
    def __init__(
        self,
        config: DynamicTransformerConfig,
        *,
        mod_dim: int = 4,
        mod_scale: float = 1.0,
    ) -> None:
        super().__init__(
            config,
            placement="branch_pre_activation",
            mod_dim=mod_dim,
            mod_scale=mod_scale,
        )


class GatePreActivationDirectMod(SharedAttentionUniqueFFNDirectMod):
    def __init__(
        self,
        config: DynamicTransformerConfig,
        *,
        mod_dim: int = 4,
        mod_scale: float = 1.0,
    ) -> None:
        super().__init__(
            config,
            placement="gate_pre_activation",
            mod_dim=mod_dim,
            mod_scale=mod_scale,
        )


class PostActivationDirectMod(SharedAttentionUniqueFFNDirectMod):
    def __init__(
        self,
        config: DynamicTransformerConfig,
        *,
        mod_dim: int = 4,
        mod_scale: float = 1.0,
    ) -> None:
        super().__init__(
            config,
            placement="post_activation",
            mod_dim=mod_dim,
            mod_scale=mod_scale,
        )
