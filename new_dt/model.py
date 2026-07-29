from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import DynamicTransformerConfig
from .layers import (
    DynamicTransformerBlock,
    SharedRMSNorm,
    TokenRoutedEmbedding,
    TokenRoutedLMHead,
)
from .pools import RoutedParameterTensor


@dataclass(slots=True)
class DynamicTransformerOutput:
    logits: Tensor
    loss: Tensor | None = None


def _sinusoidal_positions(max_seq_len: int, dim: int) -> Tensor:
    positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / max(dim, 1))
    )
    table = torch.zeros(max_seq_len, dim)
    table[:, 0::2] = torch.sin(positions * frequencies)
    if dim > 1:
        table[:, 1::2] = torch.cos(positions * frequencies[: table[:, 1::2].shape[1]])
    return table


class DynamicTransformer(nn.Module):
    """Reference implementation of a token-owned scalar-pool Transformer.

    Knowledge-bearing parameters are routed by token identity. Residual streams,
    normalization, causal masking, softmax, positional encoding, and layer layout
    remain common so tokens can communicate in one shared representation space.
    """

    def __init__(self, config: DynamicTransformerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = TokenRoutedEmbedding(config)
        self.register_buffer(
            "position_encoding",
            _sinusoidal_positions(config.max_seq_len, config.d_model),
            persistent=True,
        )
        self.layers = nn.ModuleList(
            DynamicTransformerBlock(config, index)
            for index in range(config.n_layers)
        )
        self.final_norm = SharedRMSNorm(config.d_model)
        self.lm_head = TokenRoutedLMHead(config)

    def forward(
        self,
        input_ids: Tensor,
        *,
        labels: Tensor | None = None,
        collect_route_grads: bool = False,
    ) -> DynamicTransformerOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        if input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size:
            raise ValueError("input_ids contain token IDs outside the vocabulary")

        seq_len = input_ids.shape[1]
        x = self.embedding(
            input_ids, collect_route_grads=collect_route_grads
        ) + self.position_encoding[:seq_len].to(input_ids.device)
        for layer in self.layers:
            x = layer(x, input_ids, collect_route_grads=collect_route_grads)
        x = self.final_norm(x)
        logits = self.lm_head(x, collect_route_grads=collect_route_grads)

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
            )
        return DynamicTransformerOutput(logits=logits, loss=loss)

    def routed_tensors(self) -> Iterator[tuple[str, RoutedParameterTensor]]:
        for name, module in self.named_modules():
            if isinstance(module, RoutedParameterTensor):
                yield name, module

    def pool_summary(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for name, routed in self.routed_tensors():
            summary[name] = {
                "active": routed.pool.active_count,
                "capacity": routed.pool.capacity,
                "route_slots_per_token": routed.route_size,
            }
        return summary
