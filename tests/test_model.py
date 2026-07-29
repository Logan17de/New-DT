from __future__ import annotations

import torch

from new_dt import (
    DynamicStructureController,
    DynamicTransformer,
    DynamicTransformerConfig,
    RouteLocation,
)
from new_dt.layers import SharedRMSNorm
from new_dt.structure import OwnerGradientStat


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


def test_reverse_map_tracks_same_scalar_at_different_slots() -> None:
    model = DynamicTransformer(tiny_config())
    routed = model.embedding.parameters_by_token
    source = int(routed.route_ids[0, 0].item())
    replaced = int(routed.route_ids[1, 3].item())

    assert routed.reroute_slot(1, 3, source, expected_old_index=replaced)
    locations = set(routed.route_locations(source))
    assert RouteLocation(0, 0) in locations
    assert RouteLocation(1, 0) in locations
    assert RouteLocation(1, 3) in locations
    assert routed.usage_count(source) == len(locations)
    assert routed.usage_count(replaced) == 0


def test_split_reroutes_one_exact_slot_and_copies_adam_state() -> None:
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
    old_slot_one = int(routed.route_ids[0, 1].item())
    assert routed.reroute_slot(0, 1, source, expected_old_index=old_slot_one)
    new_index = routed.pool.split(source, optimizer=optimizer)
    assert routed.reroute_slot(0, 1, new_index, expected_old_index=source)

    assert int(routed.route_ids[0, 0].item()) == source
    assert int(routed.route_ids[0, 1].item()) == new_index
    assert int(routed.route_ids[1, 0].item()) == source
    assert torch.equal(routed.pool.values[source], routed.pool.values[new_index])

    state = optimizer.state[routed.pool.values]
    assert torch.equal(state["exp_avg"][source], state["exp_avg"][new_index])
    assert torch.equal(state["exp_avg_sq"][source], state["exp_avg_sq"][new_index])


def test_gradient_collection_keeps_slots_separate_even_when_pool_grad_cancels() -> None:
    model = DynamicTransformer(tiny_config())
    routed = model.embedding.parameters_by_token
    source = int(routed.route_ids[0, 0].item())
    old_slot_one = int(routed.route_ids[0, 1].item())
    assert routed.reroute_slot(0, 1, source, expected_old_index=old_slot_one)

    values = routed(torch.tensor([0]), collect_route_grads=True)
    loss = values[0, 0] - values[0, 1]
    loss.backward()
    assert torch.isclose(routed.pool.values.grad[source], torch.tensor(0.0))

    controller = DynamicStructureController(enable_merge=False, min_owner_samples=1)
    controller.collect(model)
    module_name = "embedding.parameters_by_token"
    assert controller.stats[(module_name, 0, 0, source)].ema_gradient == 1.0
    assert controller.stats[(module_name, 0, 1, source)].ema_gradient == -1.0


def test_controller_splits_only_conflicting_route_slot() -> None:
    model = DynamicTransformer(tiny_config())
    routed = model.embedding.parameters_by_token
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    source = int(routed.route_ids[0, 0].item())
    old_slot_one = int(routed.route_ids[0, 1].item())
    assert routed.reroute_slot(0, 1, source, expected_old_index=old_slot_one)

    controller = DynamicStructureController(
        structure_interval=1,
        min_owner_samples=1,
        min_gradient_magnitude=0.01,
        min_conflict_score=0.5,
        owner_threshold_scale=0.0,
        max_splits_per_pass=1,
        enable_merge=False,
    )
    module_name = "embedding.parameters_by_token"
    controller.stats[(module_name, 0, 0, source)] = OwnerGradientStat(1.0, 1.0, 4)
    controller.stats[(module_name, 0, 1, source)] = OwnerGradientStat(-1.0, 1.0, 4)
    controller.stats[(module_name, 1, 0, source)] = OwnerGradientStat(1.0, 1.0, 4)
    controller._affected_scalars[module_name].add(source)

    events = controller.maybe_restructure(
        model, optimizer, optimizer_step=1, force=True
    )
    assert len(events) == 1
    event = events[0]
    assert event.kind == "split"
    assert event.token_id == 0
    assert event.route_slot == 1
    assert int(routed.route_ids[0, 0].item()) == source
    assert int(routed.route_ids[0, 1].item()) == event.target_scalar
    assert int(routed.route_ids[1, 0].item()) == source


def test_merge_uses_reverse_map_and_only_affected_neighborhood() -> None:
    model = DynamicTransformer(tiny_config())
    routed = model.embedding.parameters_by_token
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    module_name = "embedding.parameters_by_token"

    left = int(routed.route_ids[0, 2].item())
    right = int(routed.route_ids[1, 2].item())
    with torch.no_grad():
        routed.pool.values[left] = 0.123400
        routed.pool.values[right] = 0.123405

    controller = DynamicStructureController(
        structure_interval=1,
        min_owner_samples=1,
        enable_merge=True,
        merge_weight_tolerance=1e-4,
        merge_gradient_tolerance=1e-3,
        merge_min_samples=1,
        max_merges_per_pass=1,
        max_splits_per_pass=0,
    )
    controller.stats[(module_name, 0, 2, left)] = OwnerGradientStat(0.02, 0.02, 3)
    controller.stats[(module_name, 1, 2, right)] = OwnerGradientStat(0.0201, 0.0201, 3)
    controller._affected_scalars[module_name].add(left)
    controller._optimizer_active_scalars[module_name].update({left, right})

    events = controller.maybe_restructure(
        model, optimizer, optimizer_step=1, force=True
    )
    assert len(events) == 1
    event = events[0]
    assert event.kind == "merge"
    assert routed.usage_count(event.source_scalar) == 0
    assert not bool(routed.pool.active_mask[event.source_scalar])
    assert routed.usage_count(event.target_scalar) >= 2
