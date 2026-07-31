import torch

from new_dt import DynamicTransformerConfig
from new_dt.all_models_dashboard_cli import (
    MODEL_ORDER,
    build_model,
    estimate_parameter_counts,
)
from new_dt.comparison import CorpusBundle, make_batch_plan
from new_dt.small_gpt import SharedFFN, SharedSelfAttention
from new_dt.small_lookup_dt import LookupFFN, LookupSelfAttention
from new_dt.small_unique_attn_ffn_mod import (
    SharedFFNWithPostActivationTokenMod,
    UniqueAttentionSharedFFNMod,
)
from new_dt.word_tokenizer import WordSpaceTokenizer


def config() -> DynamicTransformerConfig:
    return DynamicTransformerConfig(
        vocab_size=24,
        d_model=8,
        n_heads=2,
        n_layers=3,
        ffn_dim=16,
        max_seq_len=8,
        dropout=0.0,
    )


def test_all_five_parameter_estimates_match_instantiated_models() -> None:
    cfg = config()
    estimates = estimate_parameter_counts(cfg, mod_dim=2)
    for name in MODEL_ORDER:
        model = build_model(name, cfg, mod_dim=2, mod_scale=1.0)
        actual = sum(parameter.numel() for parameter in model.parameters())
        assert estimates[name] == actual


def test_each_model_uses_the_declared_attention_and_ffn_families() -> None:
    cfg = config()
    models = {
        name: build_model(name, cfg, mod_dim=2, mod_scale=1.0)
        for name in MODEL_ORDER
    }

    for layer in models["gpt"].layers:
        assert isinstance(layer.attention, SharedSelfAttention)
        assert isinstance(layer.ffn, SharedFFN)

    for layer in models["direct_dt"].layers:
        assert isinstance(layer.attention, LookupSelfAttention)
        assert isinstance(layer.ffn, LookupFFN)

    for layer in models["shared_attn_unique_ffn"].layers:
        assert isinstance(layer.attention, SharedSelfAttention)
        assert isinstance(layer.ffn, LookupFFN)

    for layer in models["unique_attn_shared_ffn"].layers:
        assert isinstance(layer.attention, LookupSelfAttention)
        assert isinstance(layer.ffn, SharedFFN)

    mod_model = models["unique_attn_shared_ffn_mod"]
    assert isinstance(mod_model, UniqueAttentionSharedFFNMod)
    for layer in mod_model.layers:
        assert isinstance(layer.attention, LookupSelfAttention)
        assert isinstance(layer.ffn, SharedFFNWithPostActivationTokenMod)


def test_existing_mod_has_one_table_and_one_projection_per_layer() -> None:
    cfg = config()
    model = UniqueAttentionSharedFFNMod(cfg, mod_dim=2)
    assert model.token_mod.weight.shape == (cfg.vocab_size, 2)
    projection_ids = {
        id(layer.ffn.modifier.projection.weight)
        for layer in model.layers
    }
    assert len(projection_ids) == cfg.n_layers
    assert all(
        not hasattr(layer.ffn.modifier, "token_mod")
        for layer in model.layers
    )


def test_batch_plan_is_deterministic_and_reusable() -> None:
    tokenizer = WordSpaceTokenizer(
        tokens=["<pad>", "<bos>", "<eos>", "<unk>", "a", "b"],
        lowercase=False,
    )
    corpus = CorpusBundle(
        tokenizer=tokenizer,
        train_tokens=torch.arange(128, dtype=torch.long) % 6,
        validation_tokens=torch.arange(64, dtype=torch.long) % 6,
        source_characters=0,
    )
    first = make_batch_plan(
        corpus,
        steps=20,
        grad_accum=1,
        batch_size=4,
        eval_batches=5,
        seq_len=8,
        seed=43,
    )
    second = make_batch_plan(
        corpus,
        steps=20,
        grad_accum=1,
        batch_size=4,
        eval_batches=5,
        seq_len=8,
        seed=43,
    )
    assert torch.equal(first.train_starts, second.train_starts)
    assert torch.equal(first.validation_starts, second.validation_starts)
