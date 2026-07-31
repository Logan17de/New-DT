from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from . import all_models_dashboard_cli as base


def train_model_with_hard_overfit_stop(
    name,
    config,
    corpus,
    plan,
    *,
    args,
    run_dir: Path,
    device: torch.device,
    state,
    dashboard,
) -> dict[str, Any]:
    """Train one direct-MOD model and stop when val PPL is >5% above best.

    This is intentionally scoped to the direct-MOD placement runner. The normal
    five-model benchmark keeps its original full-length behavior.
    """

    base.common._set_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = base.build_model(
        name,
        config,
        mod_dim=args.ffn_mod_dim,
        mod_scale=args.ffn_mod_scale,
    ).to(device)
    if not bool(model.lm_is_untied):
        raise RuntimeError("embedding and LM head unexpectedly share storage")

    optimizers, sparse_parameters, dense_parameters = base._build_optimizers(
        name, model, args
    )
    total_parameters, parameter_bytes = base._parameter_stats(model)
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

    tracker = base.OverfitTracker(
        relative_ppl_threshold=args.overfit_threshold,
        patience=args.overfit_patience,
    )
    initial_loss, initial_ppl = base.common.evaluate(
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
    completed_steps = 0
    early_stopped = False
    early_stop_step: int | None = None
    early_stop_regression_percent = float("nan")

    for step_index in range(args.steps):
        model.train()
        learning_rate = base.common._learning_rate(
            step_index,
            total_steps=args.steps,
            base_lr=args.lr,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        optimizers.set_lr(learning_rate)

        base.common._sync(device)
        started = time.perf_counter()
        micro_losses: list[float] = []
        for micro_index in range(args.grad_accum):
            batch = base.common.materialize_batch(
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
            sparse_gradient_norm = base._clip_gradients(
                sparse_parameters, args.grad_clip
            )
            dense_gradient_norm = base._clip_gradients(
                dense_parameters, args.grad_clip
            )
        else:
            sparse_gradient_norm = 0.0
            dense_gradient_norm = base._clip_gradients(
                dense_parameters, args.grad_clip
            )

        optimizers.step()
        optimizers.zero_grad()
        base.common._sync(device)
        train_seconds += time.perf_counter() - started

        step = step_index + 1
        completed_steps = step
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
            train_seconds / step * (args.steps - step) if step > 0 else float("nan")
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
                    "train_ppl": base._safe_ppl(final_train_loss),
                    "learning_rate": learning_rate,
                    "tokens_per_second": tokens_per_second,
                    "sparse_gradient_norm": sparse_gradient_norm,
                    "dense_gradient_norm": dense_gradient_norm,
                }
            )

        evaluation_due = step % args.eval_interval == 0 or step == args.steps
        if evaluation_due:
            last_validation_loss, last_validation_ppl = base.common.evaluate(
                model,
                corpus.validation_tokens,
                plan.validation_starts,
                seq_len=config.max_seq_len,
                device=device,
            )
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

            regression = (
                last_validation_ppl / tracker.best_ppl - 1.0
                if math.isfinite(tracker.best_ppl) and tracker.best_ppl > 0
                else float("nan")
            )
            hard_stop = (
                step > tracker.best_step
                and math.isfinite(regression)
                and regression > args.hard_stop_overfit_ratio
            )
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
                    "regression_from_best_ratio": regression,
                    "hard_stop_triggered": hard_stop,
                    "hard_stop_threshold_ratio": args.hard_stop_overfit_ratio,
                }
            )
            base._save_dashboard_state(
                run_dir, dashboard.states, dashboard.selected
            )
            if hard_stop:
                early_stopped = True
                early_stop_step = step
                early_stop_regression_percent = regression * 100.0

        if dashboard_due or evaluation_due:
            dashboard.update()

        if step % args.static_log_interval == 0 or step == args.steps:
            if device.type == "cuda":
                state.peak_gpu_bytes = int(torch.cuda.max_memory_allocated(device))
            dashboard.snapshot(model=name, step=step)
            base._save_dashboard_state(run_dir, dashboard.states, dashboard.selected)

        if early_stopped:
            state.status = "complete"
            dashboard.snapshot(model=name, step=step)
            dashboard.update()
            break

    state.validation_loss = last_validation_loss
    state.validation_ppl = last_validation_ppl
    state.best_ppl = tracker.best_ppl
    state.best_step = tracker.best_step
    state.overfit_step = tracker.onset_step
    state.status = "complete"
    state.eta_seconds = 0.0
    if device.type == "cuda":
        state.peak_gpu_bytes = int(torch.cuda.max_memory_allocated(device))
    dashboard.update()

    summary: dict[str, Any] = {
        "model": name,
        "label": base.MODEL_LABELS[name],
        "architecture": base.MODEL_ARCHITECTURES[name],
        "initial_validation_loss": initial_loss,
        "initial_validation_ppl": initial_ppl,
        "final_train_loss": final_train_loss,
        "final_train_ppl": base._safe_ppl(final_train_loss),
        "final_validation_loss": last_validation_loss,
        "final_validation_ppl": last_validation_ppl,
        "best_validation_loss": tracker.best_loss,
        "best_validation_ppl": tracker.best_ppl,
        "best_step": tracker.best_step,
        "overfit_detected": tracker.overfit_detected,
        "overfit_onset_step": tracker.onset_step,
        "overfit_patience_evaluations": args.overfit_patience,
        "overfit_relative_ppl_threshold": args.overfit_threshold,
        "final_vs_best_ppl_regression_percent": tracker.final_regression_percent(
            last_validation_ppl
        ),
        "early_stopped": early_stopped,
        "early_stop_step": early_stop_step,
        "early_stop_reason": (
            f"validation PPL exceeded best PPL by more than "
            f"{args.hard_stop_overfit_ratio * 100:.2f}%"
            if early_stopped
            else None
        ),
        "early_stop_regression_percent": early_stop_regression_percent,
        "hard_stop_overfit_ratio": args.hard_stop_overfit_ratio,
        "requested_steps": args.steps,
        "completed_steps": completed_steps,
        "steps_after_best": completed_steps - tracker.best_step,
        "processed_target_tokens": processed_tokens,
        "training_seconds": train_seconds,
        "tokens_per_second": processed_tokens / max(train_seconds, 1e-9),
        "total_parameters": total_parameters,
        "trainable_parameters": total_parameters,
        "parameter_bytes": parameter_bytes,
        "estimated_fp32_training_state_bytes": state.estimated_training_bytes,
        "peak_gpu_allocated_bytes": state.peak_gpu_bytes,
        "generalization_gap_loss": last_validation_loss - final_train_loss,
        "sample": None,
    }
    if hasattr(model, "lookup_summary"):
        summary.update(model.lookup_summary())

    base._write_json(model_dir / "summary.json", summary)
    if not args.no_save_checkpoint:
        torch.save(
            {
                "model": model.state_dict(),
                "config": asdict(config),
                "tokenizer_tokens": corpus.tokenizer.tokens,
                "summary": summary,
                "checkpoint_note": (
                    "This checkpoint is from the final completed step, which may "
                    "be an early-stop step. Use best_step for best validation PPL."
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
