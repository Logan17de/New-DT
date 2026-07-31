import math

from new_dt.direct_ffn_mod_variants_training_cli import (
    PLACEMENTS,
    _ensure_hard_stop,
    _extract_placement,
    _lock_mod_dim_four,
)


def test_placement_selection_strips_custom_flag() -> None:
    placement, remaining = _extract_placement(
        ["--placement", "gate", "--steps", "30000"]
    )
    assert placement == "gate"
    assert remaining == ["--steps", "30000"]
    assert PLACEMENTS[placement] == "direct-mod-gate-pre"


def test_post_placement_accepts_equals_form() -> None:
    placement, remaining = _extract_placement(
        ["--placement=post", "--device", "cuda"]
    )
    assert placement == "post"
    assert remaining == ["--device", "cuda"]


def test_mod_dimension_is_locked_to_four() -> None:
    arguments = _lock_mod_dim_four(["--steps", "10"])
    index = arguments.index("--ffn-mod-dim")
    assert arguments[index + 1] == "4"


def test_hard_stop_defaults_to_five_percent() -> None:
    arguments = _ensure_hard_stop(["--steps", "10"])
    index = arguments.index("--hard-stop-overfit-ratio")
    assert math.isclose(float(arguments[index + 1]), 0.05)


def test_explicit_hard_stop_is_preserved() -> None:
    arguments = _ensure_hard_stop(
        ["--hard-stop-overfit-ratio=0.07", "--steps", "10"]
    )
    assert arguments.count("--hard-stop-overfit-ratio=0.07") == 1
    assert "--hard-stop-overfit-ratio" not in arguments
