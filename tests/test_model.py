from __future__ import annotations

import torch

from new_dt import (
    DynamicStructureController,
    DynamicTransformer,
    DynamicTransformerConfig,
    RouteLocation,
)
from new_dt.layers import RotaryEmbedding, SharedRMSNorm
from new_dt.structure import OwnerGradientStat


def tiny_config(**overrides: object) -> DynamicTransformerConfig:
    values: dict[str, object] = {
        "vocab_size": 8,
        "d_model": 4,
        "n_heads": 2,
        "n_layers": 1,
        "ffn_dim": 8,
        "max_seq_len": 8,
        "initial_shared_fraction": 0.5,
        "pool_growth_factor": 2.0,
        "route_page_size": 4,
        "route_templates_per_page": 2,
        "route_delta_promotion_threshold": 3,
        "route_template_promotion_threshold": 4,
        "route_template_promotion_fraction": 1.0,
    }
    values.update(overrides)
    return DynamicTransformerConfig(**values)  # type: ignore[arg-type]


def test_forward_backward_and_adam() -> None:
    model = DynamicTransformer(tiny_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    input_ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    output = model(input_ids, labels=input_ids, collect_route_grads=True)
    assert output.logits.shape == (2, 4, 8)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    optimizer.step()


def test_pool_families_are_separate_norm_is_shared_and_rope_is_used() -> None:
    model = DynamicTransformer(tiny_config())
    pools = {name: routed.pool for name, routed in model.routed_tensors()}
    assert "embedding.parameters_by_token" in pools
    assert "lm_head.parameters_by_token" in pools
    assert any("attention" in name for name in pools)
    assert any("ffn" in name for name in pools)
    assert len({id(pool.values) for pool in pools.values()}) == len(pools)
    assert isinstance(model.layers[0].attention_norm, SharedRMSNorm)
    assert isinstance(model.layers[0].ffn_norm, SharedRMSNorm)
    assert isinstance(model.layers[0].attention.rope, RotaryEmbedding)
    assert not hasattr(model, "position_encoding")


def test_reverse_map_tracks_same_scalar_at_different_slots() -> None:
    model = DynamicTransformer(tiny_config())
    routed = model.embedding.parameters_by_token
    source = int(routed.route_ids[0, 0].item())
    replaced = int(routed.route_ids[1, 3].item())
    replaced_before = routed.usage_count(replaced)

    assert routed.reroute_slot(1, 3, source, expected_old_index=replaced)
    locations = set(routed.route_locations(source))
    assert RouteLocation(0, 0) in locations
    assert RouteLocation(1, 0) in locations
    assert RouteLocation(1, 3) in locations
    assert routed.usage_count(source) == len(locations)
    assert routed.usage_count(replaced) == replaced_before - 1


def test_split_is_an_exception_and_template_stays_immutable() -> None:
    model = DynamicTransformer(tiny_config())
    routed = model.embedding.parameters_by_token
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    input_ids = torch.tensor([[0, 1, 2, 3]])
    output = model(input_ids, labels=input_ids)
    assert output.loss is not None
    output.loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    recipe_before = routed.page_recipe(0, 0)
    template_before = routed.route_program._templates[0][recipe_before.template_id].clone()
    source = routed.scalar_at(RouteLocation(0, 0))
    new_index = routed.pool.split(source, optimizer=optimizer)
    assert routed.reroute_slot(0, 0, new_index, expected_old_index=source)

    recipe_after = routed.page_recipe(0, 0)
    assert recipe_after.template_id == recipe_before.template_id
    assert recipe_after.exception_count == 1
    assert torch.equal(
        routed.route_program._templates[0][recipe_before.template_id], template_before
    )
    assert routed.scalar_at(RouteLocation(0, 0)) == new_index
    assert routed.scalar_at(RouteLocation(1, 0)) == source

    state = optimizer.state[routed.pool.values]
    assert torch.equal(state["exp_avg"][source], state["exp_avg"][new_index])
    assert torch.equal(state["exp_avg_sq"][source], state["exp_avg_sq"][new_index])


def test_repeated_exceptions_promote_to_shared_delta() -> None:
    model = DynamicTransformer(
        tiny_config(
            route_delta_promotion_threshold=1,
            route_template_promotion_threshold=99,
            route_template_promotion_fraction=1.0,
            route_shared_delta_min_reuse=2,
        )
    )
    routed = model.embedding.parameters_by_token
    source = routed.scalar_at(RouteLocation(0, 1))
    target = routed.pool.split(source)

    assert routed.reroute_slot(0, 1, target, expected_old_index=source)
    assert routed.page_recipe(0, 0).delta_id is None
    assert routed.reroute_slot(1, 1, target, expected_old_index=source)

    first = routed.page_recipe(0, 0)
    second = routed.page_recipe(1, 0)
    assert first.delta_id is not None
    assert first.delta_id == second.delta_id
    assert first.exception_count == 0
    assert second.exception_count == 0
    assert routed.scalar_at(RouteLocation(0, 1)) == target
    assert routed.scalar_at(RouteLocation(1, 1)) == target


def test_large_delta_is_absorbed_into_new_immutable_template() -> None:
    model = DynamicTransformer(
        tiny_config(
            route_delta_promotion_threshold=99,
            route_template_promotion_threshold=2,
            route_template_promotion_fraction=0.5,
        )
    )
    routed = model.embedding.parameters_by_token
    original_recipe = routed.page_recipe(0, 0)
    original_template = routed.route_program._templates[0][
        original_recipe.template_id
    ].clone()

    old_zero = routed.scalar_at(RouteLocation(0, 0))
    old_one = routed.scalar_at(RouteLocation(0, 1))
    target_zero = routed.scalar_at(RouteLocation(0, 2))
    target_one = routed.scalar_at(RouteLocation(0, 3))
    assert routed.reroute_slot(0, 0, target_zero, expected_old_index=old_zero)
    assert routed.reroute_slot(0, 1, target_one, expected_old_index=old_one)

    promoted = routed.page_recipe(0, 0)
    assert promoted.template_id != original_recipe.template_id
    assert promoted.delta_id is None
    assert promoted.exception_count == 0
    assert torch.equal(
        routed.route_program._templates[0][original_recipe.template_id],
        original_template,
    )
    assert routed.scalar_at(RouteLocation(0, 0)) == target_zero
    assert routed.scalar_at(RouteLocation(0, 1)) == target_one


def test_page_resolution_and_storage_estimate_are_selective() -> None:
    model = DynamicTransformer(tiny_config())
    routed = model.layers[0].ffn.up_proj.parameters_by_token
    page = routed.resolve_page(0, 1)
    full = routed.route_program.resolve_token(0)
    start = routed.route_program.page_size
    assert torch.equal(page.cpu(), full[start : start + page.numel()])

    estimate = routed.routing_storage_estimate()
    dense_int64_bytes = routed.vocab_size * routed.route_size * 8
    assert estimate["total_bytes"] < dense_int64_bytes
    assert estimate["selector_bits"] > 0
    assert estimate["template_bits"] > 0


def test_gradient_collection_keeps_slots_separate_even_when_pool_grad_cancels() -> None:
    model = DynamicTransformer(tiny_config())
    routed = model.embedding.parameters_by_token
    source = routed.scalar_at(RouteLocation(0, 0))
    old_slot_one = routed.scalar_at(RouteLocation(0, 1))
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
    source = routed.scalar_at(RouteLocation(0, 0))
    old_slot_one = routed.scalar_at(RouteLocation(0, 1))
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
    assert routed.scalar_at(RouteLocation(0, 0)) == source
    assert routed.scalar_at(RouteLocation(0, 1)) == event.target_scalar
    assert routed.scalar_at(RouteLocation(1, 0)) == source


def test_merge_uses_program_reverse_index_and_compacts_routes() -> None:
    model = DynamicTransformer(tiny_config())
    routed = model.embedding.parameters_by_token
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    module_name = "embedding.parameters_by_token"

    left_location = RouteLocation(0, 2)
    right_location = RouteLocation(4, 2)
    left = routed.scalar_at(left_location)
    right = routed.scalar_at(right_location)
    assert left != right
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
    controller.stats[(module_name, 4, 2, right)] = OwnerGradientStat(
        0.0201, 0.0201, 3
    )
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
