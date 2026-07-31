from __future__ import annotations

from argparse import Namespace

import torch

from new_dt.config import DynamicTransformerConfig
from new_dt.hybrid_comparison import (
    _build_optimizers,
    estimate_parameter_counts,
)
from new_dt.small_gpt import SharedFFN, SharedSelfAttention
from new_dt.small_hybrid_dt import (
    SharedAttentionUniqueFFN,
    UniqueAttentionSharedFFN,
)
from new_dt.small_lookup_dt import LookupFFN, LookupSelfAttention


def _config() -> DynamicTransformerConfig:
    return DynamicTransformerConfig(
        vocab_size=12,
        d_model=8,
        n_heads=2,
        n_layers=1,
        ffn_dim=16,
        max_seq_len=8,
        dropout=0.0,
    )


def _optimizer_args() -> Namespace:
    return Namespace(
        lr=3e-4,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.0,
    )


def test_shared_attention_unique_ffn_structure_and_forward() -> None:
    config = _config()
    model = SharedAttentionUniqueFFN(config)
    block = model.layers[0]

    assert isinstance(block.attention, SharedSelfAttention)
    assert isinstance(block.ffn, LookupFFN)
    assert model.lm_is_untied

    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    output = model(tokens, labels=tokens)
    assert output.logits.shape == (2, 4, config.vocab_size)
    assert output.loss is not None


def test_unique_attention_shared_ffn_structure_and_forward() -> None:
    config = _config()
    model = UniqueAttentionSharedFFN(config)
    block = model.layers[0]

    assert isinstance(block.attention, LookupSelfAttention)
    assert isinstance(block.ffn, SharedFFN)
    assert model.lm_is_untied

    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    output = model(tokens, labels=tokens)
    assert output.logits.shape == (2, 4, config.vocab_size)
    assert output.loss is not None


def test_only_unique_component_uses_sparse_lookup_gradients() -> None:
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    shared_attention = SharedAttentionUniqueFFN(_config())
    output = shared_attention(tokens, labels=tokens)
    assert output.loss is not None
    output.loss.backward()
    assert shared_attention.embedding.weight.grad is not None
    assert shared_attention.embedding.weight.grad.is_sparse
    assert shared_attention.layers[0].ffn.up_proj.lookup.weight.grad is not None
    assert shared_attention.layers[0].ffn.up_proj.lookup.weight.grad.is_sparse
    assert shared_attention.layers[0].attention.q_proj.weight.grad is not None
    assert not shared_attention.layers[0].attention.q_proj.weight.grad.is_sparse

    unique_attention = UniqueAttentionSharedFFN(_config())
    output = unique_attention(tokens, labels=tokens)
    assert output.loss is not None
    output.loss.backward()
    assert unique_attention.embedding.weight.grad is not None
    assert unique_attention.embedding.weight.grad.is_sparse
    assert unique_attention.layers[0].attention.q_proj.lookup.weight.grad is not None
    assert unique_attention.layers[0].attention.q_proj.lookup.weight.grad.is_sparse
    assert unique_attention.layers[0].ffn.up_proj.weight.grad is not None
    assert not unique_attention.layers[0].ffn.up_proj.weight.grad.is_sparse


def test_hybrid_parameter_estimates_match_constructed_models() -> None:
    config = _config()
    estimates = estimate_parameter_counts(config)

    shared_attention = SharedAttentionUniqueFFN(config)
    unique_attention = UniqueAttentionSharedFFN(config)

    assert sum(p.numel() for p in shared_attention.parameters()) == estimates[
        "shared_attn_unique_ffn"
    ]
    assert sum(p.numel() for p in unique_attention.parameters()) == estimates[
        "unique_attn_shared_ffn"
    ]


def test_hybrid_optimizer_updates_sparse_and_dense_groups() -> None:
    model = SharedAttentionUniqueFFN(_config())
    optimizers, sparse_parameters, dense_parameters = _build_optimizers(
        model,
        _optimizer_args(),
    )
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    before_sparse = model.layers[0].ffn.up_proj.lookup.weight.detach().clone()
    before_dense = model.layers[0].attention.q_proj.weight.detach().clone()

    output = model(tokens, labels=tokens)
    assert output.loss is not None
    output.loss.backward()
    assert sparse_parameters
    assert dense_parameters
    optimizers.step()

    assert not torch.equal(
        model.layers[0].ffn.up_proj.lookup.weight.detach(),
        before_sparse,
    )
    assert not torch.equal(
        model.layers[0].attention.q_proj.weight.detach(),
        before_dense,
    )
