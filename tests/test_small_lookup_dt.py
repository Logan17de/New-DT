from __future__ import annotations

import pytest
import torch

from new_dt.config import DynamicTransformerConfig
from new_dt.lookup_comparison import (
    _build_optimizers,
    _clip_gradients,
    _validate_args,
    build_parser,
    estimate_parameter_counts,
)
from new_dt.small_lookup_dt import SmallLookupDT, TokenLookupLinear


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


def test_lookup_dt_forward_is_untied_and_has_no_routing_controller() -> None:
    model = SmallLookupDT(_config())
    input_ids = torch.tensor([[1, 2, 3, 2, 4, 5, 2, 6]], dtype=torch.long)

    output = model(input_ids, labels=input_ids)

    assert output.logits.shape == (1, 8, 12)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert model.lm_is_untied
    assert not hasattr(model, "compact_routes")
    assert not hasattr(model, "routed_tensors")


def test_token_owned_tables_receive_sparse_active_row_gradients() -> None:
    model = SmallLookupDT(_config())
    input_ids = torch.tensor([[1, 2, 3, 2, 4, 5, 2, 6]], dtype=torch.long)
    output = model(input_ids, labels=input_ids)
    assert output.loss is not None
    output.loss.backward()

    active = set(input_ids.unique().tolist())
    assert model.embedding.weight.grad is not None
    assert model.embedding.weight.grad.is_sparse
    embedding_rows = set(
        model.embedding.weight.grad.coalesce().indices()[0].tolist()
    )
    assert embedding_rows.issubset(active)

    lookup_modules = [
        module for module in model.modules() if isinstance(module, TokenLookupLinear)
    ]
    assert len(lookup_modules) == 7
    for module in lookup_modules:
        gradient = module.lookup.weight.grad
        assert gradient is not None
        assert gradient.is_sparse
        rows = set(gradient.coalesce().indices()[0].tolist())
        assert rows.issubset(active)

    assert model.lm_head.weight.grad is not None
    assert not model.lm_head.weight.grad.is_sparse


def test_sparse_and_dense_adam_step_lookup_dt() -> None:
    config = _config()
    model = SmallLookupDT(config)
    parser = build_parser()
    args = parser.parse_args(
        [
            "--data",
            "unused.txt",
            "--model",
            "dt",
            "--d-model",
            "8",
            "--heads",
            "2",
            "--layers",
            "1",
            "--ffn-dim",
            "16",
            "--seq-len",
            "8",
        ]
    )
    _validate_args(args)
    optimizers = _build_optimizers("dt", model, args)

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    before = model.embedding.weight.detach().clone()
    output = model(input_ids, labels=input_ids)
    assert output.loss is not None
    output.loss.backward()
    norm = _clip_gradients(model.parameters(), 1.0)
    optimizers.step()
    optimizers.zero_grad()

    assert norm > 0
    assert not torch.equal(before[1:9], model.embedding.weight.detach()[1:9])
    assert torch.equal(before[0], model.embedding.weight.detach()[0])


def test_lookup_parameter_estimate_matches_constructed_model() -> None:
    config = _config()
    model = SmallLookupDT(config)
    counts = estimate_parameter_counts(config)

    actual = sum(parameter.numel() for parameter in model.parameters())
    assert counts["dt"] == actual
    assert counts["dt"] > counts["gpt"]
    assert counts["dt_token_matrix_parameters_per_layer"] == 640


def test_lookup_parser_has_safe_defaults_and_rejects_structure_updates() -> None:
    parser = build_parser()
    args = parser.parse_args(["--data", "unused.txt"])
    assert args.model == "both"
    assert args.d_model == 16
    assert args.layers == 1
    assert args.ffn_dim == 32
    assert args.structure_interval == 0
    _validate_args(args)

    dynamic_args = parser.parse_args(
        ["--data", "unused.txt", "--structure-interval", "100"]
    )
    with pytest.raises(ValueError, match="no split/merge"):
        _validate_args(dynamic_args)
