import math

import torch

from new_dt import DynamicTransformerConfig
from new_dt.all_models_dashboard_cli import (
    MODEL_ORDER,
    build_model,
    build_parser,
    detect_overfitting,
    estimate_parameter_counts,
)


def config() -> DynamicTransformerConfig:
    return DynamicTransformerConfig(
        vocab_size=16,
        d_model=8,
        n_heads=2,
        n_layers=3,
        ffn_dim=16,
        max_seq_len=8,
        dropout=0.0,
    )


def test_dashboard_defaults_match_requested_benchmark() -> None:
    parser = build_parser()
    args = parser.parse_args(["--data", "dummy.txt"])
    assert args.model == "all"
    assert args.d_model == 32
    assert args.layers == 3
    assert args.ffn_dim == 64
    assert args.steps == 30_000
    assert args.static_log_interval == 5_000
    assert args.dashboard_interval == 10


def test_parameter_estimates_include_one_shared_mod_table() -> None:
    cfg = config()
    counts = estimate_parameter_counts(cfg, mod_dim=2)

    vocabulary = cfg.vocab_size
    width = cfg.d_model
    ffn = cfg.ffn_dim
    layers = cfg.n_layers
    embedding_head = 2 * vocabulary * width
    norms = layers * 2 * width + width
    attention = 4 * width * width
    shared_ffn = 3 * width * ffn

    expected_unique_shared = (
        embedding_head
        + layers * (vocabulary * attention + shared_ffn)
        + norms
    )
    expected_mod = (
        expected_unique_shared
        + vocabulary * 2
        + layers * 2 * ffn
    )
    assert counts["unique_attn_shared_ffn"] == expected_unique_shared
    assert counts["unique_attn_shared_ffn_mod"] == expected_mod


def test_overfit_onset_requires_sustained_validation_regression() -> None:
    evaluations = [
        (0, math.log(20.0), 20.0, float("nan")),
        (50, math.log(10.0), 10.0, math.log(9.0)),
        (100, math.log(9.0), 9.0, math.log(8.0)),
        (150, math.log(9.1), 9.1, math.log(7.8)),
        (200, math.log(9.2), 9.2, math.log(7.6)),
        (250, math.log(9.3), 9.3, math.log(7.4)),
    ]
    tracker = detect_overfitting(
        evaluations,
        relative_ppl_threshold=0.005,
        patience=3,
    )
    assert tracker.best_step == 100
    assert tracker.best_ppl == 9.0
    assert tracker.overfit_detected
    assert tracker.onset_step == 150


def test_one_noisy_validation_point_is_not_overfitting() -> None:
    evaluations = [
        (0, math.log(20.0), 20.0, float("nan")),
        (50, math.log(10.0), 10.0, math.log(9.0)),
        (100, math.log(9.0), 9.0, math.log(8.0)),
        (150, math.log(9.2), 9.2, math.log(7.8)),
        (200, math.log(8.9), 8.9, math.log(7.6)),
    ]
    tracker = detect_overfitting(
        evaluations,
        relative_ppl_threshold=0.005,
        patience=3,
    )
    assert tracker.best_step == 200
    assert not tracker.overfit_detected
    assert tracker.onset_step is None


def test_all_five_models_build_and_run() -> None:
    cfg = config()
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    for name in MODEL_ORDER:
        model = build_model(
            name,
            cfg,
            mod_dim=2,
            mod_scale=1.0,
        )
        output = model(tokens, labels=tokens)
        assert output.logits.shape == (
            2,
            cfg.max_seq_len,
            cfg.vocab_size,
        )
        assert output.loss is not None
        assert model.lm_is_untied
