from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch
from rich.text import Text

from . import all_models_dashboard_cli as base
from .config import DynamicTransformerConfig
from .small_shared_attn_mod_unique_ffn import SharedAttentionModUniqueFFN


MODEL_NAME = "shared_attn_mod_unique_ffn"
CLI_NAME = "shared-attn-mod-unique-ffn"
MODEL_LABEL = "Shared ATTN + token ATTN MOD + unique FFN"
MODEL_ARCHITECTURE = (
    "shared Q/K/V/O attention plus one cross-layer-shared token Attention MOD "
    "table with layer-specific d_model projections, followed by token-unique "
    "SwiGLU Up/Gate/Down matrices"
)

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
    return int(
        baseline
        + config.vocab_size * mod_dim
        + config.n_layers * mod_dim * config.d_model
    )


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
    dense = (
        head
        + layers * (shared_attention + mod_dim * width)
        + norms
    )
    return int(sparse * 12 + dense * 16)


class SingleModelDashboard(_ORIGINAL_DASHBOARD):
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
            f"[bold]Controlled paired Attention-MOD run[/bold]  "
            f"d={self.config.d_model} · layers={self.config.n_layers} · "
            f"heads={self.config.n_heads} · FFN={self.config.ffn_dim} · "
            f"steps={self.args.steps:,}\n"
            f"{current}  |  elapsed {elapsed}  |  {gpu}"
        )


def _install_model() -> None:
    if MODEL_NAME in base.MODEL_LABELS:
        return

    base.CLI_TO_MODEL[CLI_NAME] = MODEL_NAME  # type: ignore[assignment]
    base.MODEL_LABELS[MODEL_NAME] = MODEL_LABEL  # type: ignore[index]
    base.MODEL_ARCHITECTURES[MODEL_NAME] = MODEL_ARCHITECTURE  # type: ignore[index]

    def estimate_parameter_counts(
        config: DynamicTransformerConfig,
        mod_dim: int,
    ) -> dict[Any, int]:
        result: dict[Any, int] = dict(
            _ORIGINAL_PARAMETER_COUNTS(config, mod_dim)
        )
        result[MODEL_NAME] = _model_parameter_count(config, mod_dim)
        return result

    def estimate_training_bytes(
        config: DynamicTransformerConfig,
        mod_dim: int,
    ) -> dict[Any, int]:
        result: dict[Any, int] = dict(
            _ORIGINAL_TRAINING_BYTES(config, mod_dim)
        )
        result[MODEL_NAME] = _model_training_bytes(config, mod_dim)
        return result

    def build_model(
        name: Any,
        config: DynamicTransformerConfig,
        *,
        mod_dim: int,
        mod_scale: float,
    ):
        if name == MODEL_NAME:
            return SharedAttentionModUniqueFFN(
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
            "# New-DT paired Shared-ATTN + Attention-MOD + unique-FFN run",
            1,
        )
        text += (
            "\n## Attention MOD definition\n\n"
            "The token-specific projected MOD is added to the shared attention "
            "output at model width, before the attention residual connection. "
            "It is not added inside the FFN. One token MOD table is shared across "
            "all layers, and each layer has its own d_mod-to-d_model projection.\n"
            "\n## Paired-comparison guarantee\n\n"
            "With the same seed, this model constructs the complete Shared-ATTN "
            "+ unique-FFN baseline before adding the Attention MOD. The MOD table "
            "starts at zero, so baseline parameters, step-zero logits, and "
            "step-zero loss are exactly identical.\n"
        )
        report_path.write_text(text, encoding="utf-8")

    base.estimate_parameter_counts = estimate_parameter_counts  # type: ignore[assignment]
    base.estimate_training_bytes = estimate_training_bytes  # type: ignore[assignment]
    base.build_model = build_model  # type: ignore[assignment]
    base.Dashboard = SingleModelDashboard  # type: ignore[assignment]
    base._write_final_reports = write_final_reports  # type: ignore[assignment]


def _translate_attention_mod_flags(arguments: list[str]) -> list[str]:
    translated: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--attn-mod-dim":
            translated.append("--ffn-mod-dim")
        elif argument.startswith("--attn-mod-dim="):
            translated.append(
                "--ffn-mod-dim=" + argument.split("=", 1)[1]
            )
        elif argument == "--attn-mod-scale":
            translated.append("--ffn-mod-scale")
        elif argument.startswith("--attn-mod-scale="):
            translated.append(
                "--ffn-mod-scale=" + argument.split("=", 1)[1]
            )
        else:
            translated.append(argument)
        index += 1
    return translated


def main(argv: list[str] | None = None) -> int:
    _install_model()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(
        argument == "--model" or argument.startswith("--model=")
        for argument in arguments
    ):
        raise SystemExit(
            "This command always runs Shared ATTN + token Attention MOD + "
            "unique FFN; remove --model."
        )
    arguments = _translate_attention_mod_flags(arguments)
    return base.main(["--model", CLI_NAME, *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
