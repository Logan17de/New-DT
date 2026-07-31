from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import torch
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


class TokenLowRankModifier(nn.Module):
    """Token-specific low-rank additive projection B_t(A_t x).

    A and B are stored as sparse embedding rows, so only tokens present in the
    current batch receive modifier gradients. B starts at zero, preserving the
    shared projection exactly at initialization while A starts random.
    """

    def __init__(
        self,
        vocab_size: int,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        init_std: float,
        scale: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("modifier rank must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = float(scale)
        self.a = nn.Embedding(vocab_size, rank * in_features, sparse=True)
        self.b = nn.Embedding(vocab_size, out_features * rank, sparse=True)
        nn.init.normal_(
            self.a.weight,
            mean=0.0,
            std=init_std / math.sqrt(max(in_features, 1)),
        )
        nn.init.zeros_(self.b.weight)

    def forward(self, x: Tensor, token_ids: Tensor) -> Tensor:
        a = self.a(token_ids).view(*token_ids.shape, self.rank, self.in_features)
        b = self.b(token_ids).view(*token_ids.shape, self.out_features, self.rank)
        hidden = torch.einsum("...ri,...i->...r", a, x)
        return self.scale * torch.einsum("...or,...r->...o", b, hidden)

    @property
    def parameters_per_token(self) -> int:
        return self.rank * (self.in_features + self.out_features)


class SharedLinearWithTokenMod(nn.Module):
    def __init__(
        self,
        config: DynamicTransformerConfig,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        scale: float,
    ) -> None:
        super().__init__()
        self.shared = SharedLinear(
            in_features,
            out_features,
            init_std=config.init_std,
        )
        self.modifier = TokenLowRankModifier(
            config.vocab_size,
            in_features,
            out_features,
            rank,
            init_std=config.init_std,
            scale=scale,
        )

    def forward(self, x: Tensor, token_ids: Tensor) -> Tensor:
        return self.shared(x) + self.modifier(x, token_ids)


class SharedFFNWithTokenMod(nn.Module):
    def __init__(
        self,
        config: DynamicTransformerConfig,
        *,
        rank: int,
        scale: float,
    ) -> None:
        super().__init__()
        self.up_proj = SharedLinearWithTokenMod(
            config,
            config.d_model,
            config.ffn_dim,
            rank=rank,
            scale=scale,
        )
        self.gate_proj = SharedLinearWithTokenMod(
            config,
            config.d_model,
            config.ffn_dim,
            rank=rank,
            scale=scale,
        )
        self.down_proj = SharedLinearWithTokenMod(
            config,
            config.ffn_dim,
            config.d_model,
            rank=rank,
            scale=scale,
        )

    def forward(self, x: Tensor, token_ids: Tensor) -> Tensor:
        up = self.up_proj(x, token_ids)
        gate = self.gate_proj(x, token_ids)
        return self.down_proj(F.silu(gate) * up, token_ids)


class UniqueAttentionFFNModBlock(nn.Module):
    def __init__(
        self,
        config: DynamicTransformerConfig,
        *,
        mod_rank: int,
        mod_scale: float,
    ) -> None:
        super().__init__()
        self.attention_norm = SharedRMSNorm(config.d_model)
        self.attention = LookupSelfAttention(config)
        self.ffn_norm = SharedRMSNorm(config.d_model)
        self.ffn = SharedFFNWithTokenMod(
            config,
            rank=mod_rank,
            scale=mod_scale,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, token_ids: Tensor) -> Tensor:
        x = x + self.dropout(self.attention(self.attention_norm(x), token_ids))
        return x + self.dropout(self.ffn(self.ffn_norm(x), token_ids))


class UniqueAttentionSharedFFNMod(nn.Module):
    """Unique token attention plus shared FFN with small token low-rank MODs."""

    def __init__(
        self,
        config: DynamicTransformerConfig,
        *,
        mod_rank: int = 4,
        mod_scale: float = 1.0,
    ) -> None:
        super().__init__()
        config.validate()
        if mod_rank <= 0:
            raise ValueError("mod_rank must be positive")
        self.config = config
        self.mod_rank = int(mod_rank)
        self.mod_scale = float(mod_scale)
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            sparse=True,
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=config.init_std)
        self.layers = nn.ModuleList(
            UniqueAttentionFFNModBlock(
                config,
                mod_rank=self.mod_rank,
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
        for module in self.modules():
            if isinstance(module, TokenLookupLinear):
                yield module.lookup.weight
            elif isinstance(module, TokenLowRankModifier):
                yield module.a.weight
                yield module.b.weight

    def dense_parameters(self) -> Iterator[nn.Parameter]:
        sparse_ids = {id(parameter) for parameter in self.sparse_parameters()}
        for parameter in self.parameters():
            if id(parameter) not in sparse_ids:
                yield parameter

    def lookup_summary(self) -> dict[str, int | float | str]:
        sparse = list(self.sparse_parameters())
        dense = list(self.dense_parameters())
        attention_per_token = 4 * self.config.d_model * self.config.d_model
        mod_per_token = 3 * self.mod_rank * (
            self.config.d_model + self.config.ffn_dim
        )
        return {
            "architecture": "unique_attn_shared_ffn_mod",
            "mod_rank": self.mod_rank,
            "mod_scale": self.mod_scale,
            "unique_attention_parameters_per_token_per_layer": attention_per_token,
            "ffn_mod_parameters_per_token_per_layer": mod_per_token,
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
        return UniqueAttentionFFNModOutput(logits=logits, loss=loss)
