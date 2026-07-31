from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch
from rich.text import Text

from . import all_models_dashboard_cli as base
from .config import DynamicTransformerConfig
from .small_direct_ffn_mod_variants import (
    BranchPreActivationDirectMod,
    GatePreActivationDirectMod,
    PostActivationDirectMod,
)


BRANCH_MODEL = "direct_mod_branch_pre"
GATE_MODEL = "direct_mod_gate_pre"
POST_MODEL = "direct_mod_post"
MODEL_ORDER = (BRANCH_MODEL, GATE_MODEL, POST_MODEL)

MODEL_LABELS = {
    BRANCH_MODEL: "Direct MOD-4 · branch before activation",
    GATE_MODEL: "Direct MOD-4 · gate before activation",
    POST_MODEL: "Direct MOD-4 · after activation",
}

MODEL_ARCHITECTURES = {
    BRANCH_MODEL: (
        "shared attention, token-unique FFN, and one cross-layer token MOD-4 "
        "table tiled directly to FFN width and added to the Up/branch before "
        "SwiGLU multiplication"
    ),
    GATE_MODEL: (
        "shared attention, token-unique FFN, and one cross-layer token MOD-4 "
        "table tiled directly to FFN width and added to the Gate before SiLU"
    ),
    POST_MODEL: (
        "shared attention, token-unique FFN, and one cross-layer token MOD-4 "
        "table tiled directly to FFN width and added after SwiGLU activation "
        "before the token-unique Down projection"
    ),
}

_ORIGINAL_PARAMETER_COUNTS = base.estimate_parameter_counts
_ORIGINAL_TRAINING_BYTES = base.estimate_training_bytes
_ORIGINAL_BUILD_MODEL = base.build_model
_ORIGINAL_DASHBOARD = base.Dashboard
_ORIGINAL_WRITE_REPORTS = base._write_final_reports


def _model_parameter_count(
    config: DynamicTransformerConfig,
    mod_dim: int,
) -> int:
    baseline = _ORIGINAL_PARAMETER_COUNTS(config, mod_dim)[
        "shared_attn_unique_ffn"
    ]
    return int(baseline + config.vocab_size * mod_dim)


def _model_training_bytes(
    config: DynamicTransformerConfig,
    mod_dim: int,
) -> int:
    vocabulary = config.vocab_size
    width = config.d_model
    ffn = config.ffn_dim
    layers = config.n_layers

    embedding = vocabulary * width
    head = vocabulary * width
    norms = layers * 2 * width + width
    shared_attention = 4 * width * width
    unique_ffn = 3 * width * ffn

    sparse = (
        embedding
        + layers * vocabulary * unique_ffn
        + vocabulary * mod_dim
    )
    dense = head + layers * shared_attention + norms
    return int(sparse * 12 + dense * 16)


class DirectModTrioDashboard(_ORIGINAL_DASHBOARD):
    def _header(self) -> Text:
        running = next(
            (
                self.states[name]
                for name in self.selected
                if self.states[name].status == "training"
            ),
            None,
        )
        elapsed = base._format_duration(time.perf_counter() - self.started)
        gpu = "CPU"
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            gpu = (
                f"CUDA allocated {base._format_bytes(allocated)} · "
                f"reserved {base._format_bytes(reserved)}"
            )
        current = (
            "initializing"
            if running is None
            else f"{running.label} · ETA {base._format_duration(running.eta_seconds)}"
        )
        return Text.from_markup(
            f"[bold]Controlled direct MOD-4 placement benchmark[/bold]  "
            f"d={self.config.d_model} · layers={self.config.n_layers} · "
            f"heads={self.config.n_heads} · FFN={self.config.ffn_dim} · "
            f"steps={self.args.steps:,}\n"
            f"{current}  |  elapsed {elapsed}  |  {gpu}"
        )


def _install_models() -> None:
    if BRANCH_MODEL in base.MODEL_LABELS:
        return

    # The dedicated command intentionally makes `--model all` mean these three
    # variants only. It does not modify the original five-model CLI process.
    base.MODEL_ORDER = MODEL_ORDER  # type: ignore[assignment]
    for name in MODEL_ORDER:
        cli_name = name.replace("_", "-")
        base.CLI_TO_MODEL[cli_name] = name  # type: ignore[assignment]
        base.MODEL_LABELS[name] = MODEL_LABELS[name]  # type: ignore[index]
        base.MODEL_ARCHITECTURES[name] = MODEL_ARCHITECTURES[name]  # type: ignore[index]

    def estimate_parameter_counts(
        config: DynamicTransformerConfig,
        mod_dim: int,
    ) -> dict[Any, int]:
        result: dict[Any, int] = dict(
            _ORIGINAL_PARAMETER_COUNTS(config, mod_dim)
        )
        count = _model_parameter_count(config, mod_dim)
        for name in MODEL_ORDER:
            result[name] = count
        return result

    def estimate_training_bytes(
        config: DynamicTransformerConfig,
        mod_dim: int,
    ) -> dict[Any, int]:
        result: dict[Any, int] = dict(
            _ORIGINAL_TRAINING_BYTES(config, mod_dim)
        )
        size = _model_training_bytes(config, mod_dim)
        for name in MODEL_ORDER:
            result[name] = size
        return result

    def build_model(
        name: Any,
        config: DynamicTransformerConfig,
        *,
        mod_dim: int,
        mod_scale: float,
    ):
        classes = {
            BRANCH_MODEL: BranchPreActivationDirectMod,
            GATE_MODEL: GatePreActivationDirectMod,
            POST_MODEL: PostActivationDirectMod,
        }
        model_class = classes.get(name)
        if model_class is not None:
            return model_class(
                config,
                mod_dim=mod_dim,
                mod_scale=mod_scale,
            )
        return _ORIGINAL_BUILD_MODEL(
            name,
            config,
            mod_dim=mod_dim,
            mod_scale=mod_scale,
        )

    def write_final_reports(
        run_dir: Path,
        results: list[dict[str, Any]],
        *,
        config: DynamicTransformerConfig,
        args,
    ) -> None:
        _ORIGINAL_WRITE_REPORTS(
            run_dir,
            results,
            config=config,
            args=args,
        )
        report_path = run_dir / "report.md"
        text = report_path.read_text(encoding="utf-8")
        text = text.replace(
            "# New-DT controlled all-model benchmark",
            "# New-DT direct MOD-4 placement benchmark",
            1,
        )
        text += (
            "\n## Direct MOD definition\n\n"
            "All three models use one zero-initialized token MOD table shared "
            "across all Transformer layers. Each token owns exactly four values. "
            "There is no learned projection matrix. The four values are tiled "
            "periodically to the FFN width, then added at the selected placement.\n"
            "\nFor FFN width 64: `m0,m1,m2,m3` is repeated sixteen times.\n"
            "\n## Paired-comparison guarantee\n\n"
            "Every variant constructs the complete Shared-ATTN + unique-FFN "
            "baseline before adding its zero-initialized MOD table. With the same "
            "seed, the three variants and the prior baseline have identical base "
            "parameters, step-zero logits, and step-zero loss. The only trained "
            "architectural difference between these three runs is MOD placement.\n"
        )
        report_path.write_text(text, encoding="utf-8")

    base.estimate_parameter_counts = estimate_parameter_counts  # type: ignore[assignment]
    base.estimate_training_bytes = estimate_training_bytes  # type: ignore[assignment]
    base.build_model = build_model  # type: ignore[assignment]
    base.Dashboard = DirectModTrioDashboard  # type: ignore[assignment]
    base._write_final_reports = write_final_reports  # type: ignore[assignment]


def _lock_mod_dim_four(arguments: list[str]) -> list[str]:
    found = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--ffn-mod-dim":
            if index + 1 >= len(arguments):
                raise SystemExit("--ffn-mod-dim requires a value")
            found = True
            try:
                value = int(arguments[index + 1])
            except ValueError as error:
                raise SystemExit("--ffn-mod-dim must be an integer") from error
            if value != 4:
                raise SystemExit(
                    "This locked comparison requires --ffn-mod-dim 4."
                )
            index += 2
            continue
        if argument.startswith("--ffn-mod-dim="):
            found = True
            try:
                value = int(argument.split("=", 1)[1])
            except ValueError as error:
                raise SystemExit("--ffn-mod-dim must be an integer") from error
            if value != 4:
                raise SystemExit(
                    "This locked comparison requires --ffn-mod-dim 4."
                )
        index += 1

    if not found:
        arguments.extend(["--ffn-mod-dim", "4"])
    return arguments


def main(argv: list[str] | None = None) -> int:
    _install_models()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(
        argument == "--model" or argument.startswith("--model=")
        for argument in arguments
    ):
        raise SystemExit(
            "This command always runs all three direct MOD-4 placements; "
            "remove --model."
        )
    arguments = _lock_mod_dim_four(arguments)
    return base.main(["--model", "all", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
