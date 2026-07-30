from __future__ import annotations

from pathlib import Path

import torch

from new_dt import (
    DynamicTransformer,
    DynamicTransformerConfig,
    PackedSPRCReader,
    pack_uints,
    unpack_uints,
)
from new_dt.pools import RouteLocation


def config(**overrides: object) -> DynamicTransformerConfig:
    values: dict[str, object] = {
        "vocab_size": 12,
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "ffn_dim": 16,
        "max_seq_len": 8,
        "initial_shared_fraction": 0.5,
        "pool_growth_factor": 2.0,
        "route_page_size": 8,
        "route_templates_per_page": 3,
        "route_delta_promotion_threshold": 2,
        "route_template_promotion_threshold": 6,
        "route_template_promotion_fraction": 0.75,
        "route_shared_delta_min_reuse": 2,
        "route_cache_pages": 8,
        "route_linear_out_tile": 2,
        "route_lm_head_tile": 3,
        "route_materialize_token_chunk": 4,
    }
    values.update(overrides)
    return DynamicTransformerConfig(**values)  # type: ignore[arg-type]


def test_arbitrary_width_integer_packing_round_trip() -> None:
    for width in (1, 3, 7, 13, 20, 31, 32):
        mask = (1 << width) - 1
        values = [0, 1 & mask, 3 & mask, mask // 2, mask]
        packed = pack_uints(values, width)
        assert unpack_uints(packed, len(values), width) == values
        assert len(packed) == (len(values) * width + 7) // 8


def test_batch_and_slice_resolution_match_full_route() -> None:
    model = DynamicTransformer(config())
    routed = model.layers[0].ffn.up_proj.parameters_by_token
    token_ids = torch.tensor([[0, 1, 0], [5, 1, 7]])
    full = routed.route_program.resolve(token_ids)
    start, stop = 5, min(routed.route_size, 29)
    sliced = routed.route_program.resolve_slice(token_ids, start, stop)
    assert torch.equal(sliced, full[..., start:stop])

    page = routed.route_program.resolve_page_batch(token_ids, 1)
    page_start = routed.route_program.page_size
    assert torch.equal(page, full[..., page_start : page_start + page.shape[-1]])


def test_program_cache_reuses_immutable_template_delta_pages() -> None:
    model = DynamicTransformer(config())
    routed = model.embedding.parameters_by_token
    routed.route_program.clear_cache()
    before = routed.route_cache_stats()
    routed.resolve_page(0, 0)
    middle = routed.route_cache_stats()
    routed.resolve_page(0, 0)
    after = routed.route_cache_stats()
    assert middle["misses"] > before["misses"]
    assert after["hits"] > middle["hits"]


def test_adaptive_selectors_do_not_allocate_vocab_times_pages_int32() -> None:
    model = DynamicTransformer(config())
    routed = model.layers[0].ffn.up_proj.parameters_by_token
    selector_state = routed.route_program.selector_state()
    defaults = selector_state["token_defaults"]
    assert torch.is_tensor(defaults)
    assert defaults.numel() == routed.vocab_size
    estimate = routed.routing_storage_estimate()
    dense_selector_bits = routed.vocab_size * routed.route_program.num_pages * 32
    assert estimate["selector_bits"] < dense_selector_bits


def test_packed_mmap_round_trip_is_exact_and_selective(tmp_path: Path) -> None:
    model = DynamicTransformer(
        config(
            route_delta_promotion_threshold=1,
            route_template_promotion_threshold=99,
            route_template_promotion_fraction=1.0,
        )
    )
    routed = model.embedding.parameters_by_token
    source = routed.scalar_at(RouteLocation(0, 1))
    target = routed.pool.split(source)
    assert routed.reroute_slot(0, 1, target, expected_old_index=source)
    assert routed.reroute_slot(1, 1, target, expected_old_index=source)

    path = tmp_path / "embedding.sprc"
    result = routed.export_packed(path)
    assert result["file_bytes"] == path.stat().st_size
    with PackedSPRCReader(path, cache_pages=4, verify_checksum=True) as reader:
        for token_id in (0, 1, 5):
            for page_id in range(routed.route_program.num_pages):
                expected = routed.route_program.resolve_page(token_id, page_id)
                actual = reader.resolve_page(token_id, page_id)
                assert torch.equal(actual, expected)
        reader.resolve_page(0, 0)
        reader.resolve_page(0, 0)
        assert reader.cache_stats()["hits"] > 0


def test_tiled_model_forward_backward_keeps_exact_gradient_routes() -> None:
    model = DynamicTransformer(config(route_linear_out_tile=1, route_lm_head_tile=2))
    ids = torch.tensor([[0, 1, 2, 3]])
    output = model(ids, labels=ids, collect_route_grads=True)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()

    routed = model.layers[0].attention.q_proj.parameters_by_token
    samples = list(routed.pop_route_gradient_samples())
    assert len(samples) == routed.parameter_shape[0]
    observed = torch.cat([sample.route_slots[0] for sample in samples])
    assert torch.equal(observed, torch.arange(routed.route_size))
