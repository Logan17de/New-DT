from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
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


class DynamicTransformer(nn.Module):
    """Token-owned scalar-pool Transformer with SPRC routes and RoPE."""

    def __init__(self, config: DynamicTransformerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = TokenRoutedEmbedding(config)
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

        # RoPE is applied to Q/K inside each attention layer. No additive
        # positional table is stored or added to the embedding stream.
        x = self.embedding(input_ids, collect_route_grads=collect_route_grads)
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

    def compact_routes(self, *, force: bool = False) -> dict[str, dict[str, int]]:
        return {
            name: routed.compact_routes(force=force)
            for name, routed in self.routed_tensors()
        }

    def routing_storage_summary(self) -> dict[str, object]:
        modules: dict[str, dict[str, int]] = {}
        total_bytes = 0
        for name, routed in self.routed_tensors():
            estimate = routed.routing_storage_estimate()
            modules[name] = estimate
            total_bytes += estimate["total_bytes"]
        return {"total_bytes": total_bytes, "modules": modules}

    def route_cache_summary(self) -> dict[str, dict[str, int]]:
        return {
            name: routed.route_cache_stats() for name, routed in self.routed_tensors()
        }

    def export_routing(self, directory: str | os.PathLike[str]) -> dict[str, object]:
        """Atomically export every routed tensor into independent packed files."""

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        modules: dict[str, dict[str, object]] = {}
        total_bytes = 0
        for name, routed in self.routed_tensors():
            filename = name.replace(".", "__") + ".sprc"
            result = routed.export_packed(target / filename)
            total_bytes += result["file_bytes"]
            modules[name] = {"file": filename, **result}

        manifest = {
            "format": "new-dt-sprc-routing",
            "version": 1,
            "config": asdict(self.config),
            "total_file_bytes": total_bytes,
            "modules": modules,
        }
        temporary = target / "manifest.json.tmp"
        final = target / "manifest.json"
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary, final)
        return manifest

    def pool_summary(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for name, routed in self.routed_tensors():
            storage = routed.routing_storage_estimate()
            cache = routed.route_cache_stats()
            summary[name] = {
                "active": routed.pool.active_count,
                "capacity": routed.pool.capacity,
                "route_slots_per_token": routed.route_size,
                "route_pages_per_token": routed.route_program.num_pages,
                "estimated_route_bytes": storage["total_bytes"],
                "neuron_id_bits": storage["neuron_id_bits"],
                "route_cache_hits": cache["hits"],
                "route_cache_misses": cache["misses"],
            }
        return summary
