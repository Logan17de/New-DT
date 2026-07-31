import torch

from new_dt import DynamicTransformerConfig
from new_dt.shared_attn_mod_unique_ffn_training_cli import (
    _model_parameter_count,
    _model_training_bytes,
    _translate_attention_mod_flags,
)
from new_dt.small_gpt import SharedSelfAttention
from new_dt.small_hybrid_dt import SharedAttentionUniqueFFN
from new_dt.small_lookup_dt import LookupFFN
from new_dt.small_shared_attn_mod_unique_ffn import (
    AttentionTokenModifier,
    SharedAttentionModUniqueFFN,
)


def config() -> DynamicTransformerConfig:
    return DynamicTransformerConfig(
        vocab_size=32,
        d_model=8,
        n_heads=2,
        n_layers=3,
        ffn_dim=16,
        max_seq_len=8,
        dropout=0.0,
    )


def test_zero_attention_mod_is_exact_paired_baseline_at_step_zero() -> None:
    cfg = config()
    torch.manual_seed(42)
    baseline = SharedAttentionUniqueFFN(cfg)
    torch.manual_seed(42)
    modified = SharedAttentionModUniqueFFN(cfg, mod_dim=2, mod_scale=1.0)

    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    baseline.eval()
    modified.eval()
    baseline_output = baseline(tokens, labels=tokens)
    modified_output = modified(tokens, labels=tokens)

    assert torch.equal(baseline_output.logits, modified_output.logits)
    assert baseline_output.loss is not None
    assert modified_output.loss is not None
    assert torch.equal(baseline_output.loss, modified_output.loss)
    assert torch.count_nonzero(modified.token_mod.weight) == 0


def test_mod_is_at_attention_and_unique_ffn_remains_unchanged() -> None:
    model = SharedAttentionModUniqueFFN(config(), mod_dim=2)
    assert model.token_mod.num_embeddings == model.config.vocab_size
    assert model.token_mod.embedding_dim == 2

    projection_ids = set()
    for layer in model.layers:
        assert isinstance(layer.attention, SharedSelfAttention)
        assert isinstance(layer.attention_modifier, AttentionTokenModifier)
        assert isinstance(layer.ffn, LookupFFN)
        projection_ids.add(id(layer.attention_modifier.projection.weight))

    assert len(projection_ids) == model.config.n_layers
    assert all(
        not hasattr(layer.attention_modifier, "token_mod")
        for layer in model.layers
    )


def test_sparse_gradients_cover_active_mod_and_unique_ffn_rows() -> None:
    model = SharedAttentionModUniqueFFN(config(), mod_dim=2)
    tokens = torch.tensor([[1, 2, 1, 3, 4, 2, 5, 6]])
    output = model(tokens, labels=tokens)
    assert output.loss is not None
    output.loss.backward()

    sparse = list(model.sparse_parameters())
    assert sparse
    assert any(parameter is model.token_mod.weight for parameter in sparse)
    assert all(parameter.grad is not None for parameter in sparse)
    assert all(parameter.grad.is_sparse for parameter in sparse)

    token_mod_gradient = model.token_mod.weight.grad.coalesce()
    active_rows = set(token_mod_gradient.indices()[0].tolist())
    assert active_rows.issubset(set(tokens.flatten().tolist()))


def test_parameter_and_training_state_estimates_match_model() -> None:
    cfg = config()
    model = SharedAttentionModUniqueFFN(cfg, mod_dim=2)
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert _model_parameter_count(cfg, 2) == actual

    sparse = sum(parameter.numel() for parameter in model.sparse_parameters())
    dense = sum(parameter.numel() for parameter in model.dense_parameters())
    assert _model_training_bytes(cfg, 2) == sparse * 12 + dense * 16


def test_mod_parameters_are_table_plus_dmodel_projection_per_layer() -> None:
    cfg = config()
    base_model = SharedAttentionUniqueFFN(cfg)
    mod_model = SharedAttentionModUniqueFFN(cfg, mod_dim=2)
    added = sum(parameter.numel() for parameter in mod_model.parameters()) - sum(
        parameter.numel() for parameter in base_model.parameters()
    )
    assert added == cfg.vocab_size * 2 + cfg.n_layers * 2 * cfg.d_model


def test_attention_mod_cli_flags_are_translated_for_shared_runner() -> None:
    assert _translate_attention_mod_flags(
        ["--attn-mod-dim", "4", "--attn-mod-scale=1.0"]
    ) == ["--ffn-mod-dim", "4", "--ffn-mod-scale=1.0"]
