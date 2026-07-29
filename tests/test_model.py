from __future__ import annotations

import torch

from new_dt import DynamicTransformer, DynamicTransformerConfig
from new_dt.layers import SharedRMSNorm


def tiny_config() -> DynamicTransformerConfig:
    return DynamicTransformerConfig(
        vocab_size=8,
        d_model=4,
        n_heads=2,
        n_layers=1,
        ffn_dim=8,
        max_seq_len=8,
        initial_shared_fraction=0.5,
        pool_growth_factor=1.5,
    )


def test_forward_backward_and_adam() -> None:
    model = DynamicTransformer(tiny_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    input_ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    output = model(input_ids, labels=input_ids, collect_route_grads=True)
    assert output.logits.shape == (2, 4, 8)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    optimizer.step()


def test_pool_families_are_separate_and_norm_is_shared() -> None:
    model = DynamicTransformer(tiny_config())
    pools = {name: routed.pool for name, routed in model.routed_tensors()}
    assert "embedding.parameters_by_token" in pools
    assert "lm_head.parameters_by_token" in pools
    assert any("attention" in name for name in pools)
    assert any("ffn" in name for name in pools)
    assert len({id(pool.values) for pool in pools.values()}) == len(pools)
    assert isinstance(model.layers[0].attention_norm, SharedRMSNorm)
    assert isinstance(model.layers[0].ffn_norm, SharedRMSNorm)


def test_two_tokens_share_half_of_embedding_route() -> None:
    model = DynamicTransformer(tiny_config())
    routed = model.embedding.parameters_by_token
    token_zero = set(routed.route_ids[0].tolist())
    token_one = set(routed.route_ids[1].tolist())
    assert len(token_zero) == 4
    assert len(token_one) == 4
    assert len(token_zero & token_one) == 2
    assert len(token_zero | token_one) == 6


def test_split_reroutes_only_selected_owner_and_copies_adam_state() -> None:
    model = DynamicTransformer(tiny_config())
    routed = model.embedding.parameters_by_token
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    input_ids = torch.tensor([[0, 1, 2, 3]])
    output = model(input_ids, labels=input_ids)
    assert output.loss is not None
    output.loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    source = int(routed.route_ids[0, 0].item())
    assert int(routed.route_ids[1, 0].item()) == source
    new_index = routed.pool.split(source, optimizer=optimizer)
    changed = routed.reroute_token_scalar(0, source, new_index)

    assert changed == 1
    assert int(routed.route_ids[0, 0].item()) == new_index
    assert int(routed.route_ids[1, 0].item()) == source
    assert torch.equal(routed.pool.values[source], routed.pool.values[new_index])

    state = optimizer.state[routed.pool.values]
    assert torch.equal(state["exp_avg"][source], state["exp_avg"][new_index])
    assert torch.equal(state["exp_avg_sq"][source], state["exp_avg_sq"][new_index])
