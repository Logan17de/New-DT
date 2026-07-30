from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
from torch import Tensor, nn

from . import comparison as shared
from .config import DynamicTransformerConfig
from .small_gpt import SmallGPT
from .small_lookup_dt import SmallLookupDT
from .word_tokenizer import WordSpaceTokenizer

ModelName = Literal["gpt", "dt"]
CorpusBundle = shared.CorpusBundle
BatchPlan = shared.BatchPlan
prepare_corpus = shared.prepare_corpus
make_batch_plan = shared.make_batch_plan
materialize_batch = shared.materialize_batch
evaluate = shared.evaluate
_resolve_device = shared._resolve_device
_learning_rate = shared._learning_rate
_set_seed = shared._set_seed
_sync = shared._sync
_generate = shared._generate


def estimate_parameter_counts(config: DynamicTransformerConfig) -> dict[str, int]:
    vocabulary = config.vocab_size
    width = config.d_model
    ffn = config.ffn_dim
    layers = config.n_layers

    embedding_and_head = 2 * vocabulary * width
    norms = layers * 2 * width + width
    shared_layer = 4 * width * width + 3 * width * ffn
    lookup_layer = vocabulary * shared_layer

    return {
        "gpt": int(embedding_and_head + layers * shared_layer + norms),
        "dt": int(embedding_and_head + layers * lookup_layer + norms),
        "dt_sparse_lookup": int(
            vocabulary * width + layers * lookup_layer
        ),
        "dt_dense": int(vocabulary * width + norms),
        "dt_token_matrix_parameters_per_layer": int(shared_layer),
    }


def estimate_training_bytes(config: DynamicTransformerConfig) -> dict[str, int]:
    counts = estimate_parameter_counts(config)
    # FP32 approximation. SparseAdam keeps dense first/second moments, but only a
    # sparse gradient for lookup rows. Dense Adam keeps parameter, gradient, and two
    # moment tensors.
    dt_bytes = counts["dt_sparse_lookup"] * 12 + counts["dt_dense"] * 16
    gpt_bytes = counts["gpt"] * 16
    return {"gpt": int(gpt_bytes), "dt": int(dt_bytes)}


def build_model(model_name: ModelName, config: DynamicTransformerConfig) -> nn.Module:
    if model_name == "gpt":
        return SmallGPT(config)
    if model_name == "dt":
        return SmallLookupDT(config)
    raise ValueError(f"unknown model: {model_name}")


def _parameter_stats(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    result = {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "parameter_bytes": int(parameter_bytes),
    }
    if hasattr(model, "lookup_summary"):
        result.update(model.lookup_summary())  # type: ignore[no-untyped-call]
    return result


def _assert_untied_head(model_name: ModelName, model: nn.Module) -> None:
    if not bool(model.lm_is_untied):  # type: ignore[attr-defined]
        raise RuntimeError(
            f"{model_name.upper()} embedding and LM head unexpectedly share storage"
        )


class OptimizerSet:
    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("at least one optimizer is required")
        self.optimizers = optimizers

    def zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def set_lr(self, learning_rate: float) -> None:
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate


def _build_optimizers(
    model_name: ModelName,
    model: nn.Module,
    args: argparse.Namespace,
) -> OptimizerSet:
    kwargs = {
        "lr": args.lr,
        "betas": (args.beta1, args.beta2),
        "eps": args.adam_eps,
    }
    if model_name == "gpt":
        return OptimizerSet(
            [
                torch.optim.AdamW(
                    model.parameters(),
                    weight_decay=args.weight_decay,
                    **kwargs,
                )
            ]
        )

    if args.weight_decay != 0.0:
        raise ValueError("lookup DT requires --weight-decay 0 for sparse row updates")
    sparse_parameters = list(model.sparse_parameters())  # type: ignore[attr-defined]
    dense_parameters = list(model.dense_parameters())  # type: ignore[attr-defined]
    return OptimizerSet(
        [
            torch.optim.SparseAdam(sparse_parameters, **kwargs),
            torch.optim.AdamW(dense_parameters, weight_decay=0.0, **kwargs),
        ]
    )


def _clip_gradients(parameters: Iterable[nn.Parameter], max_norm: float) -> float:
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters or max_norm <= 0:
        return 0.0

    device = parameters[0].device
    total_squared = torch.zeros((), device=device, dtype=torch.float32)
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse:
            gradient = gradient.coalesce()
            parameter.grad = gradient
            values = gradient.values()
            total_squared.add_(values.float().pow(2).sum())
        else:
            total_squared.add_(gradient.float().pow(2).sum())

    norm = total_squared.sqrt()
    scale = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    if float(scale) < 1.0:
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                gradient._values().mul_(scale.to(dtype=gradient.dtype))
            else:
                gradient.mul_(scale.to(dtype=gradient.dtype))
    return float(norm.detach().cpu())


def train_model(
    model_name: ModelName,
    config: DynamicTransformerConfig,
    corpus: CorpusBundle,
    plan: BatchPlan,
    *,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    _set_seed(args.seed)
    model = build_model(model_name, config).to(device)
    _assert_untied_head(model_name, model)
    optimizers = _build_optimizers(model_name, model, args)

    model_dir = run_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = model_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    initial_loss, initial_ppl = evaluate(
        model,
        corpus.validation_tokens,
        plan.validation_starts,
        seq_len=config.max_seq_len,
        device=device,
    )
    best_loss = initial_loss
    final_train_loss = float("nan")
    processed_tokens = 0
    train_seconds = 0.0

    def record(payload: dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    record(
        {
            "step": 0,
            "validation_loss": initial_loss,
            "validation_ppl": initial_ppl,
        }
    )
    print(
        f"[{model_name}] step=0 val_loss={initial_loss:.4f} "
        f"val_ppl={initial_ppl:.3f}"
    )

    optimizers.zero_grad()
    for step_index in range(args.steps):
        model.train()
        learning_rate = _learning_rate(
            step_index,
            total_steps=args.steps,
            base_lr=args.lr,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        optimizers.set_lr(learning_rate)

        _sync(device)
        started = time.perf_counter()
        micro_losses: list[float] = []
        for micro_index in range(args.grad_accum):
            starts = plan.train_starts[step_index, micro_index]
            batch = materialize_batch(
                corpus.train_tokens,
                starts,
                config.max_seq_len,
                device,
            )
            output = model(batch, labels=batch)
            if output.loss is None:
                raise RuntimeError("model did not return a training loss")
            (output.loss / args.grad_accum).backward()
            micro_losses.append(float(output.loss.detach().cpu()))

        gradient_norm = _clip_gradients(model.parameters(), args.grad_clip)
        optimizers.step()
        optimizers.zero_grad()
        _sync(device)
        train_seconds += time.perf_counter() - started

        optimizer_step = step_index + 1
        final_train_loss = sum(micro_losses) / len(micro_losses)
        processed_tokens += (
            args.batch_size * args.grad_accum * (config.max_seq_len - 1)
        )
        if optimizer_step % args.log_interval == 0 or optimizer_step == 1:
            tokens_per_second = processed_tokens / max(train_seconds, 1e-9)
            print(
                f"[{model_name}] step={optimizer_step} "
                f"train_loss={final_train_loss:.4f} lr={learning_rate:.3e} "
                f"tok/s={tokens_per_second:.1f} grad_norm={gradient_norm:.3f}"
            )
            record(
                {
                    "step": optimizer_step,
                    "train_loss": final_train_loss,
                    "learning_rate": learning_rate,
                    "tokens_per_second": tokens_per_second,
                    "gradient_norm": gradient_norm,
                }
            )

        if optimizer_step % args.eval_interval == 0 or optimizer_step == args.steps:
            validation_loss, validation_ppl = evaluate(
                model,
                corpus.validation_tokens,
                plan.validation_starts,
                seq_len=config.max_seq_len,
                device=device,
            )
            best_loss = min(best_loss, validation_loss)
            print(
                f"[{model_name}] step={optimizer_step} "
                f"val_loss={validation_loss:.4f} val_ppl={validation_ppl:.3f}"
            )
            record(
                {
                    "step": optimizer_step,
                    "validation_loss": validation_loss,
                    "validation_ppl": validation_ppl,
                }
            )

    final_loss, final_ppl = evaluate(
        model,
        corpus.validation_tokens,
        plan.validation_starts,
        seq_len=config.max_seq_len,
        device=device,
    )
    best_loss = min(best_loss, final_loss)
    sample = None
    if args.sample_prompt:
        sample = _generate(
            model,
            corpus.tokenizer,
            args.sample_prompt,
            max_new_tokens=args.sample_tokens,
            temperature=args.temperature,
            device=device,
            max_seq_len=config.max_seq_len,
        )

    summary: dict[str, Any] = {
        "model": model_name,
        "architecture": (
            "shared_matrices" if model_name == "gpt" else "direct_token_lookup"
        ),
        "structural_updates": False,
        "initial_validation_loss": initial_loss,
        "initial_validation_ppl": initial_ppl,
        "final_train_loss": final_train_loss,
        "final_validation_loss": final_loss,
        "final_validation_ppl": final_ppl,
        "best_validation_loss": best_loss,
        "best_validation_ppl": math.exp(min(best_loss, 30.0)),
        "processed_target_tokens": processed_tokens,
        "training_seconds": train_seconds,
        "tokens_per_second": processed_tokens / max(train_seconds, 1e-9),
        "sample": sample,
        **_parameter_stats(model),
    }
    (model_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not args.no_save_checkpoint:
        torch.save(
            {
                "model": model.state_dict(),
                "config": asdict(config),
                "tokenizer_tokens": corpus.tokenizer.tokens,
                "summary": summary,
            },
            model_dir / "checkpoint.pt",
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _build_config(
    args: argparse.Namespace,
    vocab_size: int,
) -> DynamicTransformerConfig:
    config = shared._build_config(args, vocab_size)
    counts = estimate_parameter_counts(config)
    if (
        args.model in ("dt", "both")
        and counts["dt"] > args.max_dt_parameters
        and not args.allow_large_dt
    ):
        memory = estimate_training_bytes(config)["dt"] / (1024**3)
        raise ValueError(
            "lookup DT is too large for the configured safety limit: "
            f"{counts['dt']:,} parameters, approximately {memory:.2f} GiB of "
            "FP32 parameters and optimizer state before activations. Reduce "
            "--d-model, --ffn-dim, or --layers, or pass --allow-large-dt."
        )
    return config


def _print_comparison(results: list[dict[str, Any]]) -> None:
    print("\nFinal comparison")
    header = (
        f"{'model':<8} {'val ppl':>10} {'best ppl':>10} {'tok/s':>12} "
        f"{'parameters':>14} {'model MB':>11} {'lookup MB':>11}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item['model']:<8} {item['final_validation_ppl']:>10.3f} "
            f"{item['best_validation_ppl']:>10.3f} "
            f"{item['tokens_per_second']:>12.1f} "
            f"{item['trainable_parameters']:>14,d} "
            f"{item['parameter_bytes'] / 1_000_000:>11.2f} "
            f"{item.get('lookup_parameter_bytes', 0) / 1_000_000:>11.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = shared.build_parser()
    parser.description = (
        "Compare exactly two untied small models on identical word-token batches: "
        "a conventional shared-matrix GPT and a direct token-lookup DT."
    )

    actions = {action.dest: action for action in parser._actions}
    actions["model"].choices = ("gpt", "dt", "both")
    actions["model"].default = "both"
    actions["d_model"].default = 16
    actions["heads"].default = 4
    actions["layers"].default = 1
    actions["ffn_dim"].default = 32
    actions["structure_interval"].default = 0
    actions["structure_interval"].help = (
        "Must be 0. Direct lookup DT has no split or merge."
    )

    parser.add_argument(
        "--max-dt-parameters",
        type=int,
        default=300_000_000,
        help="Safety limit before constructing direct lookup DT.",
    )
    parser.add_argument(
        "--allow-large-dt",
        action="store_true",
        help="Allow DT configurations above --max-dt-parameters.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    shared._validate_args(args)
    if args.structure_interval != 0:
        raise ValueError(
            "direct lookup DT has no split/merge; remove --structure-interval or set it to 0"
        )
    if args.export_routing:
        raise ValueError("direct lookup DT has no SPRC routes to export")
    if args.max_dt_parameters <= 0:
        raise ValueError("--max-dt-parameters must be positive")
    if args.model in ("dt", "both") and args.weight_decay != 0.0:
        raise ValueError("direct lookup DT requires --weight-decay 0")
