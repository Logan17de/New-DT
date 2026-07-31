from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import torch.nn.functional as F
from torch import Tensor, nn

from .config import DynamicTransformerConfig
from .layers import SharedRMSNorm
from .small_gpt import SharedFFN, SharedSelfAttention
from .small_lookup_dt import LookupFFN, LookupSelfAttention, TokenLookupLinear

HybridArchitecture = Literal[
    "shared_attn_unique_ffn",
    "unique_attn_shared_ffn",
]


@dataclass(slots=True)
class SmallHybridOutput:
    logits: Tensor
    loss: Tensor | None = None


class HybridTransformerBlock(nn.Module):
    def __init__(
        self,
        config: DynamicTransformerConfig,
        *,
        unique_attention: bool,
        unique_ffn: bool,
    ) -> None:
        super().__init__()
        if unique_attention == unique_ffn:
            raise ValueError("exactly one of attention or FFN must be token-unique")

        self.unique_attention = unique_attention
        self.unique_ffn = unique_ffn
        self.attention_norm = SharedRMSNorm(config.d_model)
        self.attention: nn.Module = (
            LookupSelfAttention(config)
            if unique_attention
            else SharedSelfAttention(config)
        )
        self.ffn_norm = SharedRMSNorm(config.d_model)
        self.ffn: nn.Module = LookupFFN(config) if unique_ffn else SharedFFN(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, token_ids: Tensor) -> Tensor:
        normalized = self.attention_norm(x)
        if self.unique_attention:
            attention_output = self.attention(normalized, token_ids)
        else:
            attention_output = self.attention(normalized)
        x = x + self.dropout(attention_output)

        normalized = self.ffn_norm(x)
        if self.unique_ffn:
            ffn_output = self.ffn(normalized, token_ids)
        else:
            ffn_output = self.ffn(normalized)
        return x + self.dropout(ffn_output)


class SmallHybridDT(nn.Module):
    """Hybrid lookup ablation with exactly one token-unique sublayer family.

    Both variants use a sparse token embedding table and a separate dense untied
    output-token table. There is no scalar pool, route sharing, SPRC, split, merge,
    compaction, or structural controller.
    """

    def __init__(
        self,
        config: DynamicTransformerConfig,
        architecture: HybridArchitecture,
    ) -> None:
        super().__init__()
        config.validate()
        if architecture not in (
            "shared_attn_unique_ffn",
            "unique_attn_shared_ffn",
        ):
            raise ValueError(f"unsupported hybrid architecture: {architecture}")

        self.config = config
        self.architecture = architecture
        self.unique_attention = architecture == "unique_attn_shared_ffn"
        self.unique_ffn = architecture == "shared_attn_unique_ffn"

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            sparse=True,
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=config.init_std)
        self.layers = nn.ModuleList(
            HybridTransformerBlock(
                config,
                unique_attention=self.unique_attention,
                unique_ffn=self.unique_ffn,
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

    def dense_parameters(self) -> Iterator[nn.Parameter]:
        sparse_ids = {id(parameter) for parameter in self.sparse_parameters()}
        for parameter in self.parameters():
            if id(parameter) not in sparse_ids:
                yield parameter

    def lookup_summary(self) -> dict[str, int | str | bool]:
        sparse = list(self.sparse_parameters())
        dense = list(self.dense_parameters())
        unique_per_token_per_layer = (
            4 * self.config.d_model * self.config.d_model
            if self.unique_attention
            else 3 * self.config.d_model * self.config.ffn_dim
        )
        return {
            "architecture": self.architecture,
            "unique_attention": self.unique_attention,
            "unique_ffn": self.unique_ffn,
            "lookup_parameters": int(sum(parameter.numel() for parameter in sparse)),
            "dense_parameters": int(sum(parameter.numel() for parameter in dense)),
            "token_owned_parameters_per_layer_per_token": int(
                unique_per_token_per_layer
            ),
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
        return SmallHybridOutput(logits=logits, loss=loss)


class SharedAttentionUniqueFFN(SmallHybridDT):
    def __init__(self, config: DynamicTransformerConfig) -> None:
        super().__init__(config, "shared_attn_unique_ffn")


class UniqueAttentionSharedFFN(SmallHybridDT):
    def __init__(self, config: DynamicTransformerConfig) -> None:
        super().__init__(config, "unique_attn_shared_ffn")
