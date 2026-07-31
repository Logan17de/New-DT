import torch

from new_dt import DynamicTransformerConfig
from new_dt.small_unique_attn_ffn_mod import (
    PostActivationTokenModifier,
    SharedFFNWithPostActivationTokenMod,
    UniqueAttentionSharedFFNMod,
)


def config(*, layers: int = 1) -> DynamicTransformerConfig:
    return DynamicTransformerConfig(
        vocab_size=32,
        d_model=8,
        n_heads=2,
        n_layers=layers,
        ffn_dim=16,
        max_seq_len=8,
        dropout=0.0,
    )


def test_zero_token_vector_preserves_shared_path() -> None:
    modifier = PostActivationTokenModifier(
        2,
        12,
        init_std=0.02,
        scale=1.0,
    )
    token_mod_vectors = torch.zeros(2, 4, 2)
    output = modifier(token_mod_vectors)
    assert output.shape == (2, 4, 12)
    assert torch.count_nonzero(output) == 0


def test_one_layer_projection_is_shared_by_all_tokens() -> None:
    modifier = PostActivationTokenModifier(
        2,
        3,
        init_std=0.02,
        scale=1.0,
    )
    with torch.no_grad():
        modifier.projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                ]
            )
        )
    token_mod_vectors = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    output = modifier(token_mod_vectors)
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
        ffn.modifier.projection.weight.zero_()
        ffn.modifier.projection.weight[0, 0] = 1.0
        ffn.down_proj.weight.zero_()
        ffn.down_proj.weight[0, 0] = 2.0
    x = torch.randn(1, 1, 8)
    token_mod_vectors = torch.tensor([[[1.0, 0.0]]])
    output = ffn(x, token_mod_vectors)
    assert output[0, 0, 0].item() == 2.0
    assert torch.count_nonzero(output[0, 0, 1:]) == 0


def test_model_uses_one_mod_table_across_layers() -> None:
    model = UniqueAttentionSharedFFNMod(config(layers=3), mod_dim=2)
    assert model.token_mod.weight.shape == (32, 2)
    assert torch.count_nonzero(model.token_mod.weight) == 0

    projection_ids = {
        id(layer.ffn.modifier.projection.weight) for layer in model.layers
    }
    assert len(projection_ids) == 3

    sparse = list(model.sparse_parameters())
    assert sum(parameter is model.token_mod.weight for parameter in sparse) == 1


def test_model_forward_and_untied_head() -> None:
    model = UniqueAttentionSharedFFNMod(config(layers=3), mod_dim=2)
    tokens = torch.randint(0, 32, (2, 8))
    output = model(tokens, labels=tokens)
    assert output.logits.shape == (2, 8, 32)
    assert output.loss is not None
    assert model.lm_is_untied


def test_sparse_modifier_gradients_accumulate_from_all_layers() -> None:
    model = UniqueAttentionSharedFFNMod(config(layers=3), mod_dim=2)
    tokens = torch.tensor([[1, 2, 1, 3, 4, 2, 5, 6]])
    output = model(tokens, labels=tokens)
    assert output.loss is not None
    output.loss.backward()
    sparse = list(model.sparse_parameters())
    assert sparse
    assert all(parameter.grad is not None for parameter in sparse)
    assert all(parameter.grad.is_sparse for parameter in sparse)
    assert model.token_mod.weight.grad is not None
    assert model.token_mod.weight.grad.is_sparse


def test_parameter_summary_counts_cross_layer_shared_table() -> None:
    model = UniqueAttentionSharedFFNMod(config(layers=3), mod_dim=2)
    summary = model.lookup_summary()
    assert summary["mod_dim"] == 2
    assert summary["ffn_mod_table_sharing"] == (
        "one_token_table_shared_across_all_layers"
    )
    assert summary["ffn_mod_parameters_per_token_total"] == 2
    assert summary["ffn_mod_projection_parameters_per_layer"] == 32
