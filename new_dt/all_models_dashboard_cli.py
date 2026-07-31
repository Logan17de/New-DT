from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from torch import nn

from . import comparison as common
from .config import DynamicTransformerConfig
from .lookup_comparison import OptimizerSet, _clip_gradients
from .small_gpt import SmallGPT
from .small_hybrid_dt import SharedAttentionUniqueFFN, UniqueAttentionSharedFFN
from .small_lookup_dt import SmallLookupDT
from .small_unique_attn_ffn_mod import UniqueAttentionSharedFFNMod
from .training_cli import (
    DEFAULT_PREPARED_ARCHIVE,
    DEFAULT_PREPARED_DIRECTORY,
    TOKENIZER_FILE,
    _default_prepared_source,
    _source_description,
    load_prepared_corpus,
)


ModelName = Literal[
    "gpt",
    "direct_dt",
    "shared_attn_unique_ffn",
    "unique_attn_shared_ffn",
    "unique_attn_shared_ffn_mod",
]

MODEL_ORDER: tuple[ModelName, ...] = (
    "gpt",
    "direct_dt",
    "shared_attn_unique_ffn",
    "unique_attn_shared_ffn",
    "unique_attn_shared_ffn_mod",
)

CLI_TO_MODEL: dict[str, ModelName] = {
    "gpt": "gpt",
    "direct-dt": "direct_dt",
    "shared-attn-unique-ffn": "shared_attn_unique_ffn",
    "unique-attn-shared-ffn": "unique_attn_shared_ffn",
    "unique-attn-shared-ffn-mod": "unique_attn_shared_ffn_mod",
}

MODEL_LABELS: dict[ModelName, str] = {
    "gpt": "Shared GPT",
    "direct_dt": "Fully unique ATTN + FFN",
    "shared_attn_unique_ffn": "Shared ATTN + unique FFN",
    "unique_attn_shared_ffn": "Unique ATTN + shared FFN",
    "unique_attn_shared_ffn_mod": "Unique ATTN + shared FFN + MOD",
}

MODEL_ARCHITECTURES: dict[ModelName, str] = {
    "gpt": "shared Q/K/V/O and shared SwiGLU FFN",
    "direct_dt": "token-unique Q/K/V/O and token-unique SwiGLU FFN",
    "shared_attn_unique_ffn": "shared Q/K/V/O and token-unique SwiGLU FFN",
    "unique_attn_shared_ffn": "token-unique Q/K/V/O and shared SwiGLU FFN",
    "unique_attn_shared_ffn_mod": (
        "token-unique Q/K/V/O, shared SwiGLU FFN, and one cross-layer-shared "
        "token MOD table with layer-specific post-activation projections"
    ),
}


@dataclass(slots=True)
class EvaluationPoint:
    step: int
    validation_loss: float
    validation_ppl: float
    train_loss: float


@dataclass(slots=True)
class OverfitTracker:
    relative_ppl_threshold: float = 0.005
    patience: int = 3
    history: list[EvaluationPoint] = field(default_factory=list)
    best_step: int = 0
    best_loss: float = float("inf")
    best_ppl: float = float("inf")
    onset_step: int | None = None

    def update(
        self,
        *,
        step: int,
        validation_loss: float,
        validation_ppl: float,
        train_loss: float,
    ) -> None:
        self.history.append(
            EvaluationPoint(
                step=step,
                validation_loss=validation_loss,
                validation_ppl=validation_ppl,
                train_loss=train_loss,
            )
        )
        best_index = min(
            range(len(self.history)),
            key=lambda index: self.history[index].validation_loss,
        )
        best = self.history[best_index]
        self.best_step = best.step
        self.best_loss = best.validation_loss
        self.best_ppl = best.validation_ppl
        self.onset_step = self._detect_onset(best_index)

    def _detect_onset(self, best_index: int) -> int | None:
        if len(self.history) < self.patience + 1:
            return None
        threshold_loss = self.best_loss + math.log1p(self.relative_ppl_threshold)
        last_start = len(self.history) - self.patience
        for start in range(best_index + 1, last_start + 1):
            window = self.history[start : start + self.patience]
            sustained_validation_worsening = all(
                point.validation_loss >= threshold_loss for point in window
            )
            if not sustained_validation_worsening:
                continue

            best_train = self.history[best_index].train_loss
            final_train = window[-1].train_loss
            training_still_improving = (
                not math.isfinite(best_train)
                or not math.isfinite(final_train)
                or final_train <= best_train
            )
            if training_still_improving:
                return window[0].step
        return None

    @property
    def overfit_detected(self) -> bool:
        return self.onset_step is not None

    def final_regression_percent(self, final_ppl: float) -> float:
        if not math.isfinite(self.best_ppl) or self.best_ppl <= 0:
            return float("nan")
        return 100.0 * (final_ppl / self.best_ppl - 1.0)


@dataclass(slots=True)
class ModelState:
    name: ModelName
    label: str
    status: str = "waiting"
    step: int = 0
    total_steps: int = 30_000
    train_loss: float = float("nan")
    validation_loss: float = float("nan")
    validation_ppl: float = float("nan")
    best_ppl: float = float("nan")
    best_step: int = 0
    overfit_step: int | None = None
    tokens_per_second: float = 0.0
    parameters: int = 0
    parameter_bytes: int = 0
    estimated_training_bytes: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: float = float("nan")
    peak_gpu_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_ppl(loss: float) -> float:
    if not math.isfinite(loss):
        return float("nan")
    return math.exp(min(loss, 30.0))


def _format_float(value: float, digits: int = 3) -> str:
    return "—" if not math.isfinite(value) else f"{value:.{digits}f}"


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "—"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _format_bytes(value: int) -> str:
    if value <= 0:
        return "—"
    gib = value / 1024**3
    if gib >= 1:
        return f"{gib:.2f} GiB"
    return f"{value / 1024**2:.1f} MiB"


def _progress_bar(step: int, total: int, width: int = 14) -> str:
    fraction = 0.0 if total <= 0 else min(max(step / total, 0.0), 1.0)
    filled = int(round(width * fraction))
    return f"[{'█' * filled}{'░' * (width - filled)}] {fraction * 100:5.1f}%"


def estimate_parameter_counts(
    config: DynamicTransformerConfig,
    mod_dim: int,
) -> dict[ModelName, int]:
    vocabulary = config.vocab_size
    width = config.d_model
    ffn = config.ffn_dim
    layers = config.n_layers

    embedding_and_head = 2 * vocabulary * width
    norms = layers * 2 * width + width
    attention = 4 * width * width
    ffn_parameters = 3 * width * ffn

    return {
        "gpt": int(
            embedding_and_head
            + layers * (attention + ffn_parameters)
            + norms
        ),
        "direct_dt": int(
            embedding_and_head
            + layers * vocabulary * (attention + ffn_parameters)
            + norms
        ),
        "shared_attn_unique_ffn": int(
            embedding_and_head
            + layers * (attention + vocabulary * ffn_parameters)
            + norms
        ),
        "unique_attn_shared_ffn": int(
            embedding_and_head
            + layers * (vocabulary * attention + ffn_parameters)
            + norms
        ),
        "unique_attn_shared_ffn_mod": int(
            embedding_and_head
            + layers
            * (
                vocabulary * attention
                + ffn_parameters
                + mod_dim * ffn
            )
            + vocabulary * mod_dim
            + norms
        ),
    }


def estimate_training_bytes(
    config: DynamicTransformerConfig,
    mod_dim: int,
) -> dict[ModelName, int]:
    vocabulary = config.vocab_size
    width = config.d_model
    ffn = config.ffn_dim
    layers = config.n_layers

    embedding = vocabulary * width
    head = vocabulary * width
    norms = layers * 2 * width + width
    attention = 4 * width * width
    ffn_parameters = 3 * width * ffn

    sparse_dense: dict[ModelName, tuple[int, int]] = {
        "gpt": (
            0,
            embedding
            + head
            + layers * (attention + ffn_parameters)
            + norms,
        ),
        "direct_dt": (
            embedding
            + layers * vocabulary * (attention + ffn_parameters),
            head + norms,
        ),
        "shared_attn_unique_ffn": (
            embedding + layers * vocabulary * ffn_parameters,
            head + layers * attention + norms,
        ),
        "unique_attn_shared_ffn": (
            embedding + layers * vocabulary * attention,
            head + layers * ffn_parameters + norms,
        ),
        "unique_attn_shared_ffn_mod": (
            embedding + layers * vocabulary * attention + vocabulary * mod_dim,
            head
            + layers * (ffn_parameters + mod_dim * ffn)
            + norms,
        ),
    }
    result: dict[ModelName, int] = {}
    for name, (sparse, dense) in sparse_dense.items():
        result[name] = int(sparse * 12 + dense * 16)
    return result


def build_model(
    name: ModelName,
    config: DynamicTransformerConfig,
    *,
    mod_dim: int,
    mod_scale: float,
) -> nn.Module:
    if name == "gpt":
        return SmallGPT(config)
    if name == "direct_dt":
        return SmallLookupDT(config)
    if name == "shared_attn_unique_ffn":
        return SharedAttentionUniqueFFN(config)
    if name == "unique_attn_shared_ffn":
        return UniqueAttentionSharedFFN(config)
    if name == "unique_attn_shared_ffn_mod":
        return UniqueAttentionSharedFFNMod(
            config,
            mod_dim=mod_dim,
            mod_scale=mod_scale,
        )
    raise ValueError(f"unknown model: {name}")


def model_names_from_args(args: argparse.Namespace) -> list[ModelName]:
    if args.model == "all":
        return list(MODEL_ORDER)
    return [CLI_TO_MODEL[args.model]]


def detect_overfitting(
    evaluations: list[tuple[int, float, float, float]],
    *,
    relative_ppl_threshold: float = 0.005,
    patience: int = 3,
) -> OverfitTracker:
    tracker = OverfitTracker(
        relative_ppl_threshold=relative_ppl_threshold,
        patience=patience,
    )
    for step, validation_loss, validation_ppl, train_loss in evaluations:
        tracker.update(
            step=step,
            validation_loss=validation_loss,
            validation_ppl=validation_ppl,
            train_loss=train_loss,
        )
    return tracker


class Dashboard:
    def __init__(
        self,
        *,
        states: dict[ModelName, ModelState],
        selected: list[ModelName],
        config: DynamicTransformerConfig,
        args: argparse.Namespace,
        device: torch.device,
    ) -> None:
        self.states = states
        self.selected = selected
        self.config = config
        self.args = args
        self.device = device
        self.console = Console()
        self.started = time.perf_counter()
        self.live: Live | None = None

    def _status_text(self, state: ModelState) -> Text:
        styles = {
            "waiting": "dim",
            "training": "bold cyan",
            "complete": "bold green",
            "failed": "bold red",
        }
        return Text(state.status.upper(), style=styles.get(state.status, ""))

    def _table(self) -> Table:
        table = Table(
            expand=True,
            show_lines=False,
            header_style="bold magenta",
            box=None,
            pad_edge=False,
        )
        table.add_column("Model", no_wrap=True, ratio=2)
        table.add_column("State", width=9)
        table.add_column("Step", justify="right", width=9)
        table.add_column("Progress", width=23)
        table.add_column("Train PPL", justify="right", width=10)
        table.add_column("Val PPL", justify="right", width=10)
        table.add_column("Best PPL @ step", justify="right", width=19)
        table.add_column("Overfit onset", justify="right", width=14)
        table.add_column("tok/s", justify="right", width=10)
        table.add_column("Params", justify="right", width=12)

        for name in self.selected:
            state = self.states[name]
            best = (
                "—"
                if not math.isfinite(state.best_ppl)
                else f"{state.best_ppl:.3f} @ {state.best_step:,}"
            )
            overfit = (
                f"{state.overfit_step:,}"
                if state.overfit_step is not None
                else ("watching" if state.status == "training" else "none")
            )
            table.add_row(
                state.label,
                self._status_text(state),
                f"{state.step:,}",
                _progress_bar(state.step, state.total_steps),
                _format_float(_safe_ppl(state.train_loss)),
                _format_float(state.validation_ppl),
                best,
                overfit,
                f"{state.tokens_per_second:,.0f}"
                if state.tokens_per_second > 0
                else "—",
                f"{state.parameters:,}" if state.parameters else "—",
            )
        return table

    def _header(self) -> Text:
        running = next(
            (self.states[name] for name in self.selected if self.states[name].status == "training"),
            None,
        )
        elapsed = _format_duration(time.perf_counter() - self.started)
        gpu = "CPU"
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            gpu = (
                f"CUDA allocated {_format_bytes(allocated)} · "
                f"reserved {_format_bytes(reserved)}"
            )
        current = (
            "initializing"
            if running is None
            else f"{running.label} · ETA {_format_duration(running.eta_seconds)}"
        )
        return Text.from_markup(
            f"[bold]Controlled 5-model benchmark[/bold]  "
            f"d={self.config.d_model} · layers={self.config.n_layers} · "
            f"heads={self.config.n_heads} · FFN={self.config.ffn_dim} · "
            f"steps={self.args.steps:,}\n"
            f"{current}  |  elapsed {elapsed}  |  {gpu}"
        )

    def render(self) -> Panel:
        footer = Text(
            "Overfit onset = first evaluation in a sustained "
            f"{self.args.overfit_patience}-evaluation run at least "
            f"{self.args.overfit_threshold * 100:.2f}% above the best PPL "
            "while training loss does not worsen.",
            style="dim",
        )
        return Panel(
            Group(self._header(), Text(""), self._table(), Text(""), footer),
            title="New-DT Live Metrics",
            border_style="blue",
        )

    def start(self) -> None:
        if self.args.no_dashboard:
            return
        self.live = Live(
            self.render(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self.live.start()

    def update(self) -> None:
        if self.live is not None:
            self.live.update(self.render(), refresh=True)

    def snapshot(self, *, model: ModelName, step: int) -> None:
        title = f"Static snapshot · {MODEL_LABELS[model]} · step {step:,}"
        panel = Panel(self._table(), title=title, border_style="yellow")
        if self.live is not None:
            self.live.console.print(panel)
        else:
            self.console.print(panel)

    def stop(self) -> None:
        if self.live is not None:
            self.live.update(self.render(), refresh=True)
            self.live.stop()
            self.live = None


def _load_corpus(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
):
    if args.data and args.prepared_data:
        parser.error("use either --data or --prepared-data, not both")

    prepared = args.prepared_data
    if not args.data and prepared is None:
        prepared = _default_prepared_source()
        if prepared is None:
            parser.error(
                "provide --prepared-data or --data; expected "
                f"{DEFAULT_PREPARED_DIRECTORY} or {DEFAULT_PREPARED_ARCHIVE}"
            )
        args.prepared_data = prepared

    if prepared is not None:
        corpus = load_prepared_corpus(
            prepared,
            validation_fraction=args.validation_fraction,
            seq_len=args.seq_len,
        )
        return corpus, _source_description(prepared)

    max_vocab = None if args.max_vocab == 0 else args.max_vocab
    corpus = common.prepare_corpus(
        args.data,
        lowercase=args.lowercase,
        min_frequency=args.min_frequency,
        max_vocab=max_vocab,
        validation_fraction=args.validation_fraction,
        seq_len=args.seq_len,
    )
    return corpus, {
        "mode": "raw_text",
        "files": [str(path) for path in args.data],
    }


def _parameter_stats(model: nn.Module) -> tuple[int, int]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    return int(parameters), int(parameter_bytes)


def _build_optimizers(
    name: ModelName,
    model: nn.Module,
    args: argparse.Namespace,
) -> tuple[
    OptimizerSet,
    list[nn.Parameter],
    list[nn.Parameter],
]:
    kwargs = {
        "lr": args.lr,
        "betas": (args.beta1, args.beta2),
        "eps": args.adam_eps,
    }
    if name == "gpt":
        dense = list(model.parameters())
        return (
            OptimizerSet(
                [
                    torch.optim.AdamW(
                        dense,
                        weight_decay=args.weight_decay,
                        **kwargs,
                    )
                ]
            ),
            [],
            dense,
        )

    if args.weight_decay != 0.0:
        raise ValueError("all token-lookup models require --weight-decay 0")
    sparse = list(model.sparse_parameters())  # type: ignore[attr-defined]
    dense = list(model.dense_parameters())  # type: ignore[attr-defined]
    return (
        OptimizerSet(
            [
                torch.optim.SparseAdam(sparse, **kwargs),
                torch.optim.AdamW(dense, weight_decay=0.0, **kwargs),
            ]
        ),
        sparse,
        dense,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _save_dashboard_state(
    run_dir: Path,
    states: dict[ModelName, ModelState],
    selected: list[ModelName],
) -> None:
    _write_json(
        run_dir / "dashboard_state.json",
        [states[name].as_dict() for name in selected],
    )


def _train_model(
    name: ModelName,
    config: DynamicTransformerConfig,
    corpus,
    plan,
    *,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    state: ModelState,
    dashboard: Dashboard,
) -> dict[str, Any]:
    common._set_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = build_model(
        name,
        config,
        mod_dim=args.ffn_mod_dim,
        mod_scale=args.ffn_mod_scale,
    ).to(device)
    if not bool(model.lm_is_untied):  # type: ignore[attr-defined]
        raise RuntimeError("embedding and LM head unexpectedly share storage")

    optimizers, sparse_parameters, dense_parameters = _build_optimizers(
        name,
        model,
        args,
    )
    total_parameters, parameter_bytes = _parameter_stats(model)
    state.parameters = total_parameters
    state.parameter_bytes = parameter_bytes
    state.status = "training"

    model_dir = run_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = model_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    def record(payload: dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    tracker = OverfitTracker(
        relative_ppl_threshold=args.overfit_threshold,
        patience=args.overfit_patience,
    )
    initial_loss, initial_ppl = common.evaluate(
        model,
        corpus.validation_tokens,
        plan.validation_starts,
        seq_len=config.max_seq_len,
        device=device,
    )
    tracker.update(
        step=0,
        validation_loss=initial_loss,
        validation_ppl=initial_ppl,
        train_loss=float("nan"),
    )
    state.validation_loss = initial_loss
    state.validation_ppl = initial_ppl
    state.best_ppl = tracker.best_ppl
    state.best_step = tracker.best_step
    record(
        {
            "event": "validation",
            "step": 0,
            "validation_loss": initial_loss,
            "validation_ppl": initial_ppl,
        }
    )
    dashboard.update()

    optimizers.zero_grad()
    processed_tokens = 0
    train_seconds = 0.0
    final_train_loss = float("nan")
    last_validation_loss = initial_loss
    last_validation_ppl = initial_ppl
    last_evaluated_step = 0

    for step_index in range(args.steps):
        model.train()
        learning_rate = common._learning_rate(
            step_index,
            total_steps=args.steps,
            base_lr=args.lr,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        optimizers.set_lr(learning_rate)

        common._sync(device)
        started = time.perf_counter()
        micro_losses: list[float] = []
        for micro_index in range(args.grad_accum):
            batch = common.materialize_batch(
                corpus.train_tokens,
                plan.train_starts[step_index, micro_index],
                config.max_seq_len,
                device,
            )
            output = model(batch, labels=batch)
            if output.loss is None:
                raise RuntimeError("model did not return a training loss")
            (output.loss / args.grad_accum).backward()
            micro_losses.append(float(output.loss.detach().cpu()))

        if sparse_parameters:
            sparse_gradient_norm = _clip_gradients(
                sparse_parameters,
                args.grad_clip,
            )
            dense_gradient_norm = _clip_gradients(
                dense_parameters,
                args.grad_clip,
            )
        else:
            sparse_gradient_norm = 0.0
            dense_gradient_norm = _clip_gradients(
                dense_parameters,
                args.grad_clip,
            )

        optimizers.step()
        optimizers.zero_grad()
        common._sync(device)
        train_seconds += time.perf_counter() - started

        step = step_index + 1
        final_train_loss = sum(micro_losses) / len(micro_losses)
        processed_tokens += (
            args.batch_size * args.grad_accum * (config.max_seq_len - 1)
        )
        tokens_per_second = processed_tokens / max(train_seconds, 1e-9)
        state.step = step
        state.train_loss = final_train_loss
        state.tokens_per_second = tokens_per_second
        state.elapsed_seconds = train_seconds
        state.eta_seconds = (
            train_seconds / step * (args.steps - step)
            if step > 0
            else float("nan")
        )

        dashboard_due = (
            step == 1
            or step % args.dashboard_interval == 0
            or step == args.steps
        )
        if dashboard_due:
            record(
                {
                    "event": "train",
                    "step": step,
                    "train_loss": final_train_loss,
                    "train_ppl": _safe_ppl(final_train_loss),
                    "learning_rate": learning_rate,
                    "tokens_per_second": tokens_per_second,
                    "sparse_gradient_norm": sparse_gradient_norm,
                    "dense_gradient_norm": dense_gradient_norm,
                }
            )

        evaluation_due = (
            step % args.eval_interval == 0 or step == args.steps
        )
        if evaluation_due:
            last_validation_loss, last_validation_ppl = common.evaluate(
                model,
                corpus.validation_tokens,
                plan.validation_starts,
                seq_len=config.max_seq_len,
                device=device,
            )
            last_evaluated_step = step
            tracker.update(
                step=step,
                validation_loss=last_validation_loss,
                validation_ppl=last_validation_ppl,
                train_loss=final_train_loss,
            )
            state.validation_loss = last_validation_loss
            state.validation_ppl = last_validation_ppl
            state.best_ppl = tracker.best_ppl
            state.best_step = tracker.best_step
            state.overfit_step = tracker.onset_step
            record(
                {
                    "event": "validation",
                    "step": step,
                    "train_loss": final_train_loss,
                    "validation_loss": last_validation_loss,
                    "validation_ppl": last_validation_ppl,
                    "best_validation_ppl": tracker.best_ppl,
                    "best_step": tracker.best_step,
                    "overfit_detected": tracker.overfit_detected,
                    "overfit_onset_step": tracker.onset_step,
                }
            )
            _save_dashboard_state(run_dir, dashboard.states, dashboard.selected)

        if dashboard_due or evaluation_due:
            dashboard.update()

        if (
            step % args.static_log_interval == 0
            or step == args.steps
        ):
            if device.type == "cuda":
                state.peak_gpu_bytes = int(
                    torch.cuda.max_memory_allocated(device)
                )
            dashboard.snapshot(model=name, step=step)
            _save_dashboard_state(run_dir, dashboard.states, dashboard.selected)

    if last_evaluated_step != args.steps:
        last_validation_loss, last_validation_ppl = common.evaluate(
            model,
            corpus.validation_tokens,
            plan.validation_starts,
            seq_len=config.max_seq_len,
            device=device,
        )
        tracker.update(
            step=args.steps,
            validation_loss=last_validation_loss,
            validation_ppl=last_validation_ppl,
            train_loss=final_train_loss,
        )

    state.validation_loss = last_validation_loss
    state.validation_ppl = last_validation_ppl
    state.best_ppl = tracker.best_ppl
    state.best_step = tracker.best_step
    state.overfit_step = tracker.onset_step
    state.status = "complete"
    if device.type == "cuda":
        state.peak_gpu_bytes = int(torch.cuda.max_memory_allocated(device))
    dashboard.update()

    sample = None
    if args.sample_prompt:
        sample = common._generate(
            model,
            corpus.tokenizer,
            args.sample_prompt,
            max_new_tokens=args.sample_tokens,
            temperature=args.temperature,
            device=device,
            max_seq_len=config.max_seq_len,
        )

    summary: dict[str, Any] = {
        "model": name,
        "label": MODEL_LABELS[name],
        "architecture": MODEL_ARCHITECTURES[name],
        "initial_validation_loss": initial_loss,
        "initial_validation_ppl": initial_ppl,
        "final_train_loss": final_train_loss,
        "final_train_ppl": _safe_ppl(final_train_loss),
        "final_validation_loss": last_validation_loss,
        "final_validation_ppl": last_validation_ppl,
        "best_validation_loss": tracker.best_loss,
        "best_validation_ppl": tracker.best_ppl,
        "best_step": tracker.best_step,
        "overfit_detected": tracker.overfit_detected,
        "overfit_onset_step": tracker.onset_step,
        "overfit_patience_evaluations": args.overfit_patience,
        "overfit_relative_ppl_threshold": args.overfit_threshold,
        "final_vs_best_ppl_regression_percent": (
            tracker.final_regression_percent(last_validation_ppl)
        ),
        "steps_after_best": args.steps - tracker.best_step,
        "processed_target_tokens": processed_tokens,
        "training_seconds": train_seconds,
        "tokens_per_second": processed_tokens / max(train_seconds, 1e-9),
        "total_parameters": total_parameters,
        "trainable_parameters": total_parameters,
        "parameter_bytes": parameter_bytes,
        "estimated_fp32_training_state_bytes": state.estimated_training_bytes,
        "peak_gpu_allocated_bytes": state.peak_gpu_bytes,
        "generalization_gap_loss": last_validation_loss - final_train_loss,
        "sample": sample,
    }
    if hasattr(model, "lookup_summary"):
        summary.update(model.lookup_summary())  # type: ignore[attr-defined]

    _write_json(model_dir / "summary.json", summary)
    if not args.no_save_checkpoint:
        torch.save(
            {
                "model": model.state_dict(),
                "config": asdict(config),
                "tokenizer_tokens": corpus.tokenizer.tokens,
                "summary": summary,
                "checkpoint_note": (
                    "This is the final-step checkpoint. The summary records the "
                    "best validation step and overfitting onset separately."
                ),
            },
            model_dir / "checkpoint.pt",
        )

    del model
    del optimizers
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _write_final_reports(
    run_dir: Path,
    results: list[dict[str, Any]],
    *,
    config: DynamicTransformerConfig,
    args: argparse.Namespace,
) -> None:
    ranked = sorted(results, key=lambda item: item["best_validation_ppl"])
    _write_json(run_dir / "comparison.json", ranked)

    csv_fields = [
        "rank",
        "model",
        "label",
        "best_validation_ppl",
        "best_step",
        "final_validation_ppl",
        "final_train_ppl",
        "overfit_detected",
        "overfit_onset_step",
        "final_vs_best_ppl_regression_percent",
        "tokens_per_second",
        "total_parameters",
        "parameter_bytes",
        "estimated_fp32_training_state_bytes",
        "peak_gpu_allocated_bytes",
        "processed_target_tokens",
    ]
    with (run_dir / "comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for rank, item in enumerate(ranked, start=1):
            writer.writerow(
                {
                    key: rank if key == "rank" else item.get(key)
                    for key in csv_fields
                }
            )

    lines = [
        "# New-DT controlled all-model benchmark",
        "",
        "## Locked configuration",
        "",
        f"- d_model: {config.d_model}",
        f"- layers: {config.n_layers}",
        f"- heads: {config.n_heads}",
        f"- FFN width: {config.ffn_dim}",
        f"- sequence length: {config.max_seq_len}",
        f"- optimizer steps per model: {args.steps:,}",
        f"- evaluation interval: {args.eval_interval:,}",
        f"- static console snapshot interval: {args.static_log_interval:,}",
        f"- MOD dimension: {args.ffn_mod_dim}",
        "",
        "## Ranking by best validation perplexity",
        "",
        "| Rank | Model | Best PPL | Best step | Final PPL | Overfit onset | Regression | Params | tok/s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(ranked, start=1):
        onset = (
            f"{item['overfit_onset_step']:,}"
            if item["overfit_onset_step"] is not None
            else "not detected"
        )
        lines.append(
            f"| {rank} | {item['label']} | "
            f"{item['best_validation_ppl']:.4f} | "
            f"{item['best_step']:,} | "
            f"{item['final_validation_ppl']:.4f} | "
            f"{onset} | "
            f"{item['final_vs_best_ppl_regression_percent']:.2f}% | "
            f"{item['total_parameters']:,} | "
            f"{item['tokens_per_second']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Overfitting rule",
            "",
            (
                "The reported onset is retrospective: the first evaluation after "
                "the best checkpoint that begins "
                f"{args.overfit_patience} consecutive evaluations at least "
                f"{args.overfit_threshold * 100:.2f}% above the best PPL, while "
                "training loss does not worsen. This avoids calling a single noisy "
                "validation point overfitting."
            ),
            "",
            "The saved checkpoint, unless disabled, is the final-step checkpoint. "
            "Use the recorded best step for a focused rerun when final quality has regressed.",
            "",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = common.build_parser()
    parser.description = (
        "Run the complete five-model New-DT comparison sequentially with one "
        "dynamic Rich dashboard and permanent snapshots every 5,000 steps."
    )
    actions = {action.dest: action for action in parser._actions}
    actions["data"].required = False
    actions["data"].help = (
        "Raw UTF-8 files. Prefer --prepared-data for the controlled SciQ run."
    )
    actions["model"].choices = (*CLI_TO_MODEL.keys(), "all")
    actions["model"].default = "all"
    actions["output_dir"].default = Path("runs/all_models_dashboard")
    actions["d_model"].default = 32
    actions["heads"].default = 4
    actions["layers"].default = 3
    actions["ffn_dim"].default = 64
    actions["seq_len"].default = 64
    actions["dropout"].default = 0.0
    actions["steps"].default = 30_000
    actions["batch_size"].default = 8
    actions["grad_accum"].default = 1
    actions["lr"].default = 3e-4
    actions["warmup_steps"].default = 50
    actions["eval_interval"].default = 50
    actions["eval_batches"].default = 20
    actions["log_interval"].default = 10
    actions["weight_decay"].default = 0.0
    actions["structure_interval"].default = 0

    parser.add_argument(
        "--prepared-data",
        type=Path,
        default=None,
        help="Directory or ZIP containing tokenizer.json and prepared tokens.",
    )
    parser.add_argument("--ffn-mod-dim", type=int, default=4)
    parser.add_argument("--ffn-mod-scale", type=float, default=1.0)
    parser.add_argument(
        "--dashboard-interval",
        type=int,
        default=10,
        help="Refresh dynamic training values every N optimizer steps.",
    )
    parser.add_argument(
        "--static-log-interval",
        type=int,
        default=5_000,
        help="Print one permanent dashboard snapshot every N optimizer steps.",
    )
    parser.add_argument(
        "--overfit-threshold",
        type=float,
        default=0.005,
        help="Relative PPL rise above the best value required for overfit detection.",
    )
    parser.add_argument(
        "--overfit-patience",
        type=int,
        default=3,
        help="Consecutive validation evaluations required to mark overfitting.",
    )
    parser.add_argument(
        "--max-parameters",
        type=int,
        default=1_000_000_000,
        help="Per-model parameter safety limit.",
    )
    parser.add_argument(
        "--allow-large-model",
        action="store_true",
        help="Permit a selected model above --max-parameters.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable live redraws while retaining 5K snapshots and result files.",
    )
    return parser


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    common._validate_args(args)
    positive = {
        "dashboard_interval": args.dashboard_interval,
        "static_log_interval": args.static_log_interval,
        "overfit_patience": args.overfit_patience,
        "ffn_mod_dim": args.ffn_mod_dim,
        "max_parameters": args.max_parameters,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.structure_interval != 0:
        parser.error("this controlled benchmark has no split/merge; use 0")
    if args.weight_decay != 0:
        parser.error("use --weight-decay 0 for the sparse lookup models")
    if args.overfit_threshold < 0:
        parser.error("--overfit-threshold cannot be negative")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    device = common._resolve_device(args.device)
    corpus, source = _load_corpus(parser, args)
    config = common._build_config(args, corpus.tokenizer.vocab_size)
    config.validate()
    selected = model_names_from_args(args)

    parameter_estimates = estimate_parameter_counts(config, args.ffn_mod_dim)
    training_estimates = estimate_training_bytes(config, args.ffn_mod_dim)
    oversized = [
        name
        for name in selected
        if parameter_estimates[name] > args.max_parameters
    ]
    if oversized and not args.allow_large_model:
        details = ", ".join(
            f"{MODEL_LABELS[name]}={parameter_estimates[name]:,}"
            for name in oversized
        )
        parser.error(
            f"model estimate exceeds --max-parameters: {details}. "
            "Increase the limit or pass --allow-large-model after checking GPU memory."
        )

    plan = common.make_batch_plan(
        corpus,
        steps=args.steps,
        grad_accum=args.grad_accum,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        seq_len=args.seq_len,
        seed=args.seed + 1,
    )
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    corpus.tokenizer.save(run_dir / TOKENIZER_FILE)

    argument_snapshot = dict(vars(args))
    argument_snapshot["data"] = (
        [str(path) for path in args.data] if args.data else None
    )
    argument_snapshot["prepared_data"] = (
        str(args.prepared_data) if args.prepared_data else None
    )
    argument_snapshot["output_dir"] = str(args.output_dir)
    _write_json(
        run_dir / "run_config.json",
        {
            "arguments": argument_snapshot,
            "model_config": asdict(config),
            "selected_models": selected,
            "architectures": {
                name: MODEL_ARCHITECTURES[name] for name in selected
            },
            "parameter_estimates": {
                name: parameter_estimates[name] for name in selected
            },
            "estimated_fp32_training_state_bytes": {
                name: training_estimates[name] for name in selected
            },
            "dataset": {
                "source": source,
                "vocab_size": corpus.tokenizer.vocab_size,
                "train_tokens": len(corpus.train_tokens),
                "validation_tokens": len(corpus.validation_tokens),
                "target_tokens_per_step": (
                    args.batch_size
                    * args.grad_accum
                    * (args.seq_len - 1)
                ),
            },
            "fairness": {
                "same_saved_tokenizer": True,
                "same_precomputed_token_stream": True,
                "same_batch_plan": True,
                "seed_reset_before_each_model": True,
                "same_dimensions_and_depth": True,
                "same_lr_schedule": True,
                "same_evaluation_batches": True,
                "sequential_training": True,
                "untied_lm_heads": True,
                "mod_table_sharing": (
                    "one token MOD table shared across all layers; "
                    "layer-specific projection matrices"
                ),
            },
        },
    )

    states = {
        name: ModelState(
            name=name,
            label=MODEL_LABELS[name],
            total_steps=args.steps,
            parameters=parameter_estimates[name],
            parameter_bytes=parameter_estimates[name] * 4,
            estimated_training_bytes=training_estimates[name],
        )
        for name in selected
    }
    dashboard = Dashboard(
        states=states,
        selected=selected,
        config=config,
        args=args,
        device=device,
    )
    dashboard.start()
    results: list[dict[str, Any]] = []
    try:
        for name in selected:
            results.append(
                _train_model(
                    name,
                    config,
                    corpus,
                    plan,
                    args=args,
                    run_dir=run_dir,
                    device=device,
                    state=states[name],
                    dashboard=dashboard,
                )
            )
            _save_dashboard_state(run_dir, states, selected)
    except Exception:
        running = next(
            (
                state
                for state in states.values()
                if state.status == "training"
            ),
            None,
        )
        if running is not None:
            running.status = "failed"
        dashboard.update()
        raise
    finally:
        dashboard.stop()

    _write_final_reports(
        run_dir,
        results,
        config=config,
        args=args,
    )

    final_table = Table(
        title="Final ranking by best validation PPL",
        header_style="bold green",
    )
    final_table.add_column("Rank", justify="right")
    final_table.add_column("Model")
    final_table.add_column("Best PPL", justify="right")
    final_table.add_column("Best step", justify="right")
    final_table.add_column("Final PPL", justify="right")
    final_table.add_column("Overfit onset", justify="right")
    final_table.add_column("Regression", justify="right")
    final_table.add_column("Parameters", justify="right")
    for rank, item in enumerate(
        sorted(results, key=lambda row: row["best_validation_ppl"]),
        start=1,
    ):
        final_table.add_row(
            str(rank),
            item["label"],
            f"{item['best_validation_ppl']:.4f}",
            f"{item['best_step']:,}",
            f"{item['final_validation_ppl']:.4f}",
            (
                f"{item['overfit_onset_step']:,}"
                if item["overfit_onset_step"] is not None
                else "not detected"
            ),
            f"{item['final_vs_best_ppl_regression_percent']:.2f}%",
            f"{item['total_parameters']:,}",
        )
    dashboard.console.print(final_table)
    dashboard.console.print(
        f"[bold green]Reports:[/bold green] {run_dir / 'report.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
