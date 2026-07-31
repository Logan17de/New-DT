import torch
import torch.nn.functional as F

from new_dt import DynamicTransformerConfig
from new_dt.direct_ffn_mod_variants_training_cli import (
    _lock_mod_dim_four,
    _model_parameter_count,
    _model_training_bytes,
)
from new_dt.small_direct_ffn_mod_variants import (
    BranchPreActivationDirectMod,
    DirectTokenModBroadcast,
    GatePreActivationDirectMod,
    PostActivationDirectMod,
    UniqueFFNWithDirectTokenMod,
)
from new_dt.small_hybrid_dt import SharedAttentionUniqueFFN
from new_dt.small_lookup_dt import LookupFFN


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


def variants():
    return (
        BranchPreActivationDirectMod,
        GatePreActivationDirectMod,
        PostActivationDirectMod,
    )


def test_fixed_broadcast_tiles_mod_four_without_parameters() -> None:
    broadcast = DirectTokenModBroadcast(4, 16, scale=1.0)
    values = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    expanded = broadcast(values)
    assert expanded.shape == (1, 1, 16)
    assert torch.equal(
        expanded,
        torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0] * 4]]
        ),
    )
    assert sum(parameter.numel() for parameter in broadcast.parameters()) == 0


def test_direct_mod_requires_exact_width_divisibility() -> None:
    try:
        DirectTokenModBroadcast(4, 18, scale=1.0)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("non-divisible direct MOD width should fail")


def test_all_three_variants_are_exact_paired_baselines_at_step_zero() -> None:
    cfg = config()
    tokens = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]]
    )

    torch.manual_seed(42)
    baseline = SharedAttentionUniqueFFN(cfg)
    baseline.eval()
    baseline_output = baseline(tokens, labels=tokens)

    for model_class in variants():
        torch.manual_seed(42)
        model = model_class(cfg, mod_dim=4, mod_scale=1.0)
        model.eval()
        output = model(tokens, labels=tokens)
        assert torch.equal(output.logits, baseline_output.logits)
        assert output.loss is not None
        assert baseline_output.loss is not None
        assert torch.equal(output.loss, baseline_output.loss)
        assert torch.count_nonzero(model.token_mod.weight) == 0
        assert not model.direct_mod_has_projection


def test_one_mod_table_is_shared_across_all_layers() -> None:
    cfg = config()
    for model_class in variants():
        model = model_class(cfg, mod_dim=4)
        assert model.token_mod.weight.shape == (cfg.vocab_size, 4)
        assert all(
            isinstance(layer.ffn, UniqueFFNWithDirectTokenMod)
            for layer in model.layers
        )
        assert all(not hasattr(layer.ffn, "token_mod") for layer in model.layers)


def test_only_token_table_parameters_are_added() -> None:
    cfg = config()
    base = SharedAttentionUniqueFFN(cfg)
    base_count = sum(parameter.numel() for parameter in base.parameters())

    for model_class in variants():
        model = model_class(cfg, mod_dim=4)
        count = sum(parameter.numel() for parameter in model.parameters())
        assert count - base_count == cfg.vocab_size * 4
        assert _model_parameter_count(cfg, 4) == count

        sparse = sum(parameter.numel() for parameter in model.sparse_parameters())
        dense = sum(parameter.numel() for parameter in model.dense_parameters())
        assert _model_training_bytes(cfg, 4) == sparse * 12 + dense * 16


def test_each_placement_matches_its_declared_equation() -> None:
    cfg = config()
    x = torch.randn(2, cfg.max_seq_len, cfg.d_model)
    token_ids = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    token_mod = torch.randn(2, cfg.max_seq_len, 4)

    placements = (
        "branch_pre_activation",
        "gate_pre_activation",
        "post_activation",
    )
    for placement in placements:
        base_ffn = LookupFFN(cfg)
        wrapped = UniqueFFNWithDirectTokenMod(
            base_ffn,
            cfg,
            mod_dim=4,
            mod_scale=1.0,
            placement=placement,
        )
        up = base_ffn.up_proj(x, token_ids)
        gate = base_ffn.gate_proj(x, token_ids)
        direct = wrapped.broadcast(token_mod)
        if placement == "branch_pre_activation":
            activated = F.silu(gate) * (up + direct)
        elif placement == "gate_pre_activation":
            activated = F.silu(gate + direct) * up
        else:
            activated = F.silu(gate) * up + direct
        expected = base_ffn.down_proj(activated, token_ids)
        actual = wrapped(x, token_ids, token_mod)
        assert torch.equal(actual, expected)


def test_sparse_gradient_reaches_only_active_mod_rows() -> None:
    model = PostActivationDirectMod(config(), mod_dim=4)
    tokens = torch.tensor([[1, 2, 1, 3, 4, 2, 5, 6]])
    output = model(tokens, labels=tokens)
    assert output.loss is not None
    output.loss.backward()

    assert model.token_mod.weight.grad is not None
    assert model.token_mod.weight.grad.is_sparse
    gradient = model.token_mod.weight.grad.coalesce()
    active_rows = set(gradient.indices()[0].tolist())
    assert active_rows.issubset(set(tokens.flatten().tolist()))


def test_locked_runner_defaults_to_mod_four_and_rejects_other_dims() -> None:
    arguments = _lock_mod_dim_four(["--steps", "10"])
    assert arguments[-2:] == ["--ffn-mod-dim", "4"]
    assert _lock_mod_dim_four(["--ffn-mod-dim", "4"]) == [
        "--ffn-mod-dim",
        "4",
    ]

    try:
        _lock_mod_dim_four(["--ffn-mod-dim=8"])
    except SystemExit as error:
        assert "requires" in str(error)
    else:
        raise AssertionError("the locked trio must reject MOD dimensions other than 4")
