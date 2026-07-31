from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from . import comparison as common
from .config import DynamicTransformerConfig
from .lookup_comparison import OptimizerSet, _clip_gradients
from .small_hybrid_dt import (
    SharedAttentionUniqueFFN,
    SmallHybridDT,
    UniqueAttentionSharedFFN,
)
from .word_tokenizer import WordSpaceTokenizer

ModelName = Literal[
    "shared_attn_unique_ffn",
    "unique_attn_shared_ffn",
]
CorpusBundle = common.CorpusBundle
BatchPlan = common.BatchPlan
prepare_corpus = common.prepare_corpus
make_batch_plan = common.make_batch_plan
materialize_batch = common.materialize_batch
evaluate = common.evaluate
_resolve_device = common._resolve_device
_learning_rate = common._learning_rate
_set_seed = common._set_seed
_sync = common._sync
_generate = common._generate

CLI_TO_MODEL: dict[str, ModelName] = {
    "shared-attn-unique-ffn": "shared_attn_unique_ffn",
    "unique-attn-shared-ffn": "unique_attn_shared_ffn",
}


def estimate_parameter_counts(config: DynamicTransformerConfig) -> dict[str, int]:
    vocabulary = config.vocab_size
    width = config.d_model
    ffn = config.ffn_dim
    layers = config.n_layers

    embedding_and_head = 2 * vocabulary * width
    norms = layers * 2 * width + width
    shared_attention = 4 * width * width
    unique_attention = vocabulary * shared_attention
    shared_ffn = 3 * width * ffn
    unique_ffn = vocabulary * shared_ffn

    shared_attn_unique_ffn = (
        embedding_and_head
        + layers * (shared_attention + unique_ffn)
        + norms
    )
    unique_attn_shared_ffn = (
        embedding_and_head
        + layers * (unique_attention + shared_ffn)
        + norms
    )

    return {
        "shared_attn_unique_ffn": int(shared_attn_unique_ffn),
        "unique_attn_shared_ffn": int(unique_attn_shared_ffn),
        "shared_attn_unique_ffn_sparse": int(
            vocabulary * width + layers * unique_ffn
        ),
        "shared_attn_unique_ffn_dense": int(
            vocabulary * width + layers * (shared_attention + 2 * width) + width
        ),
        "unique_attn_shared_ffn_sparse": int(
            vocabulary * width + layers * unique_attention
        ),
        "unique_attn_shared_ffn_dense": int(
            vocabulary * width + layers * (shared_ffn + 2 * width) + width
        ),
        "unique_attention_parameters_per_token_per_layer": int(shared_attention),
        "unique_ffn_parameters_per_token_per_layer": int(shared_ffn),
    }


def estimate_training_bytes(config: DynamicTransformerConfig) -> dict[str, int]:
    counts = estimate_parameter_counts(config)
    # FP32 estimate. SparseAdam keeps parameter plus dense first/second moments;
    # gradients remain sparse. AdamW keeps parameter, gradient, and two moments.
    return {
        "shared_attn_unique_ffn": int(
            counts["shared_attn_unique_ffn_sparse"] * 12
            + counts["shared_attn_unique_ffn_dense"] * 16
        ),
        "unique_attn_shared_ffn": int(
            counts["unique_attn_shared_ffnn_sparse"] * 12
            + counts["unique_attn_shared_ffn_dense"] * 16
        )
        if "unique_attn_shared_ffnn_sparse" in counts
        else int(
            counts["unique_attn_shared_ffn_sparse"] * 12
            + counts["unique_attn_shared_ffn_dense"] * 16
        ),
    }


def build_model(model_name: ModelName, config: DynamicTransformerConfig) -> SmallHybridDT:
    if model_name == "shared_attn_unique_ffn":
        return SharedAttentionUniqueFFN(config)
    if model_name == "unique_attn_shared_ffn":
        return UniqueAttentionSharedFFN(config)
    raise ValueError(f"unknown hybrid model: {model_name}")


def _parameter_stats(model: SmallHybridDT) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "parameter_bytes": int(
            sum(
                parameter.numel() * parameter.element_size()
                for parameter in model.parameters()
            )
        ),
        **model.lookup_summary(),
    }


def _build_optimizers(
    model: SmallHybridDT,
    args: argparse.Namespace,
) -> tuple[OptimizerSet, list[nn.Parameter], list[nn.Parameter]]:
    if args.weight_decay != 0.0:
        raise ValueError("hybrid lookup models require --weight-decay 0")

    kwargs = {
        "lr": args.lr,
        "betas": (args.beta1, args.beta2),
        "eps": args.adam_eps,
    }
    sparse_parameters = list(model.sparse_parameters())
    dense_parameters = list(model.dense_parameters())
    optimizers = OptimizerSet(
        [
            torch.optim.SparseAdam(sparse_parameters, **kwargs),
            torch.optim.AdamW(dense_parameters, weight_decay=0.0, **kwargs),
        ]
    )
    return optimizers, sparse_parameters, dense_parameters


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
    if not model.lm_is_untied:
        raise RuntimeError("embedding and LM head unexpectedly share storage")
    optimizers, sparse_parameters, dense_parameters = _build_optimizers(model, args)

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
    best_step = 0
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

        sparse_gradient_norm = _clip_gradients(sparse_parameters, args.grad_clip)
        dense_gradient_norm = _clip_gradients(dense_parameters, args.grad_clip)
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
                f"tok/s={tokens_per_second:.1f} "
                f"sparse_grad={sparse_gradient_norm:.3f} "
                f"dense_grad={dense_gradient_norm:.3f}"
            )
            record(
                {
                    "step": optimizer_step,
                    "train_loss": final_train_loss,
                    "learning_rate": learning_rate,
                    "tokens_per_second": tokens_per_second,
                    "sparse_gradient_norm": sparse_gradient_norm,
                    "dense_gradient_norm": dense_gradient_norm,
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
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_step = optimizer_step
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
    if final_loss < best_loss:
        best_loss = final_loss
        best_step = args.steps

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
        "architecture": model.architecture,
        "structural_updates": False,
        "initial_validation_loss": initial_loss,
        "initial_validation_ppl": initial_ppl,
        "final_train_loss": final_train_loss,
        "final_validation_loss": final_loss,
        "final_validation_ppl": final_ppl,
        "best_validation_loss": best_loss,
        "best_validation_ppl": math.exp(min(best_loss, 30.0)),
        "best_step": best_step,
        "processed_target_tokens": processed_tokens,
        "training_seconds": train_seconds,
        "tokens_per_second": processed_tokens / max(train_seconds, 1e-9),
        "gradient_clipping": "sparse and dense groups clipped separately",
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
    config = common._build_config(args, vocab_size)
    counts = estimate_parameter_counts(config)
    selected = (
        list(CLI_TO_MODEL.values())
        if args.model == "both"
        else [CLI_TO_MODEL[args.model]]
    )
    oversized = [
        name
        for name in selected
        if counts[name] > args.max_dt_parameters
    ]
    if oversized and not args.allow_large_dt:
        details = ", ".join(
            f"{name}={counts[name]:,}" for name in oversized
        )
        raise ValueError(
            "hybrid model exceeds --max-dt-parameters: "
            f"{details}. Reduce --d-model, --ffn-dim, or --layers, or pass "
            "--allow-large-dt after checking the hardware budget."
        )
    return config


def _print_comparison(results: list[dict[str, Any]]) -> None:
    print("\nFinal hybrid comparison")
    header = (
        f"{'model':<28} {'val ppl':>10} {'best ppl':>10} {'best step':>10} "
        f"{'tok/s':>11} {'parameters':>14} {'model MB':>10}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item['model']:<28} {item['final_validation_ppl']:>10.3f} "
            f"{item['best_validation_ppl']:>10.3f} {item['best_step']:>10,d} "
            f"{item['tokens_per_second']:>11.1f} "
            f"{item['trainable_parameters']:>14,d} "
            f"{item['parameter_bytes'] / 1_000_000:>10.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = common.build_parser()
    parser.description = (
        "Compare shared-attention/token-unique-FFN against "
        "token-unique-attention/shared-FFN on identical word-token batches."
    )
    actions = {action.dest: action for action in parser._actions}
    actions["model"].choices = (
        "shared-attn-unique-ffn",
        "unique-attn-shared-ffn",
        "both",
    )
    actions["model"].default = "both"
    actions["output_dir"].default = Path("runs/hybrid_comparison")
    actions["d_model"].default = 16
    actions["heads"].default = 4
    actions["layers"].default = 1
    actions["ffn_dim"].default = 32
    actions["structure_interval"].default = 0
    actions["structure_interval"].help = "Must be 0; hybrids have no split/merge."

    parser.add_argument(
        "--max-dt-parameters",
        type=int,
        default=300_000_000,
        help="Safety limit applied separately to each hybrid model.",
    )
    parser.add_argument(
        "--allow-large-dt",
        action="store_true",
        help="Allow hybrid models above --max-dt-parameters.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    common._validate_args(args)
    if args.structure_interval != 0:
        raise ValueError("hybrid lookup models have no split/merge")
    if args.export_routing:
        raise ValueError("hybrid lookup models have no SPRC routes")
    if args.weight_decay != 0.0:
        raise ValueError("hybrid lookup models require --weight-decay 0")
    if args.max_dt_parameters <= 0:
        raise ValueError("--max-dt-parameters must be positive")


def model_names_from_args(args: argparse.Namespace) -> list[ModelName]:
    if args.model == "both":
        return [
            "shared_attn_unique_ffn",
            "unique_attn_shared_ffn",
        ]
    return [CLI_TO_MODEL[args.model]]
