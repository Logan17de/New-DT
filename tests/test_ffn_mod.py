import torch

from new_dt import DynamicTransformerConfig
from new_dt.small_unique_attn_ffn_mod import (
    PostActivationTokenModifier,
    SharedFFNWithPostActivationTokenMod,
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
    modifier = PostActivationTokenModifier(
        16,
        2,
        12,
        init_std=0.02,
        scale=1.0,
    )
    ids = torch.randint(0, 16, (2, 4))
    output = modifier(ids)
    assert output.shape == (2, 4, 12)
    assert torch.count_nonzero(output) == 0


def test_one_common_projection_is_used_for_all_tokens() -> None:
    modifier = PostActivationTokenModifier(
        4,
        2,
        3,
        init_std=0.02,
        scale=1.0,
    )
    with torch.no_grad():
        modifier.token_mod.weight[1].copy_(torch.tensor([1.0, 0.0]))
        modifier.token_mod.weight[2].copy_(torch.tensor([0.0, 1.0]))
        modifier.projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                ]
            )
        )
    output = modifier(torch.tensor([[1, 2]]))
    assert torch.equal(output[0, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(output[0, 1], torch.tensor([10.0, 20.0, 30.0]))
    assert modifier.shared_projection_parameters == 6


def test_modifier_is_added_after_activation_before_down_projection() -> None:
    ffn = SharedFFNWithPostActivationTokenMod(
        config(),
        mod_dim=2,
        scale=1.0,
    )
    with torch.no_grad():
        ffn.up_proj.weight.zero_()
        ffn.gate_proj.weight.zero_()
        ffn.modifier.token_mod.weight[1].copy_(torch.tensor([1.0, 0.0]))
        ffn.modifier.projection.weight.zero_()
        ffn.modifier.projection.weight[0, 0] = 1.0
        ffn.down_proj.weight.zero_()
        ffn.down_proj.weight[0, 0] = 2.0
    x = torch.randn(1, 1, 8)
    output = ffn(x, torch.tensor([[1]]))
    assert output[0, 0, 0].item() == 2.0
    assert torch.count_nonzero(output[0, 0, 1:]) == 0


def test_model_forward_and_untied_head() -> None:
    model = UniqueAttentionSharedFFNMod(config(), mod_dim=2)
    tokens = torch.randint(0, 32, (2, 8))
    output = model(tokens, labels=tokens)
    assert output.logits.shape == (2, 8, 32)
    assert output.loss is not None
    assert model.lm_is_untied


def test_sparse_modifier_gradients() -> None:
    model = UniqueAttentionSharedFFNMod(config(), mod_dim=2)
    tokens = torch.tensor([[1, 2, 1, 3, 4, 2, 5, 6]])
    output = model(tokens, labels=tokens)
    assert output.loss is not None
    output.loss.backward()
    sparse = list(model.sparse_parameters())
    assert sparse
    assert all(parameter.grad is not None for parameter in sparse)
    assert all(parameter.grad.is_sparse for parameter in sparse)


def test_parameter_summary_counts_small_token_vectors() -> None:
    model = UniqueAttentionSharedFFNMod(config(), mod_dim=2)
    summary = model.lookup_summary()
    assert summary["mod_dim"] == 2
    assert summary["ffn_mod_parameters_per_token_per_layer"] == 2
    assert summary["ffn_mod_projection_parameters_per_layer"] == 32
