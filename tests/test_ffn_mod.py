import torch

from new_dt import DynamicTransformerConfig
from new_dt.small_unique_attn_ffn_mod import (
    TokenLowRankModifier,
    UniqueAttentionSharedFFNMod,
)


def config() -> DynamicTransformerConfig:
    return DynamicTransformerConfig(
        vocab_size=32,
        d_model=8,
        n_heads=2,
        n_layers=1,
        ffn_dim=16,
        max_seq_len=8,
        dropout=0.0,
    )


def test_modifier_zero_init_preserves_shared_path() -> None:
    modifier = TokenLowRankModifier(16, 8, 12, 2, init_std=0.02, scale=1.0)
    x = torch.randn(2, 4, 8)
    ids = torch.randint(0, 16, (2, 4))
    output = modifier(x, ids)
    assert torch.count_nonzero(output) == 0


def test_model_forward_and_untied_head() -> None:
    model = UniqueAttentionSharedFFNMod(config(), mod_rank=2)
    tokens = torch.randint(0, 32, (2, 8))
    output = model(tokens, labels=tokens)
    assert output.logits.shape == (2, 8, 32)
    assert output.loss is not None
    assert model.lm_is_untied


def test_sparse_modifier_gradients() -> None:
    model = UniqueAttentionSharedFFNMod(config(), mod_rank=2)
    tokens = torch.tensor([[1, 2, 1, 3, 4, 2, 5, 6]])
    output = model(tokens, labels=tokens)
    assert output.loss is not None
    output.loss.backward()
    sparse = list(model.sparse_parameters())
    assert sparse
    assert all(parameter.grad is not None for parameter in sparse)
    assert all(parameter.grad.is_sparse for parameter in sparse)


def test_parameter_summary_counts_mod_rank() -> None:
    model = UniqueAttentionSharedFFNMod(config(), mod_rank=2)
    summary = model.lookup_summary()
    assert summary["mod_rank"] == 2
    assert summary["ffn_mod_parameters_per_token_per_layer"] == 144
