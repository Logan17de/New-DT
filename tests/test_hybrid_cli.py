from pathlib import Path

import pytest

from new_dt.hybrid_training_cli import build_parser


def test_hybrid_parser_accepts_prepared_zip_and_both_models() -> None:
    args = build_parser().parse_args(
        [
            "--prepared-data",
            "data/sciq.zip",
            "--model",
            "both",
        ]
    )
    assert args.data is None
    assert args.prepared_data == Path("data/sciq.zip")
    assert args.model == "both"
    assert args.structure_interval == 0


@pytest.mark.parametrize(
    "model_name",
    [
        "shared-attn-unique-ffn",
        "unique-attn-shared-ffn",
    ],
)
def test_hybrid_parser_accepts_each_model(model_name: str) -> None:
    args = build_parser().parse_args(
        [
            "--prepared-data",
            "data/sciq.zip",
            "--model",
            model_name,
        ]
    )
    assert args.model == model_name
