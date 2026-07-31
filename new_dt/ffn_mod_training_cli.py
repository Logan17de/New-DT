from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from . import comparison as common
from .config import DynamicTransformerConfig
from .lookup_comparison import OptimizerSet, _clip_gradients
from .small_unique_attn_ffn_mod import UniqueAttentionSharedFFNMod
from .training_cli import (
    DEFAULT_PREPARED_ARCHIVE,
    DEFAULT_PREPARED_DIRECTORY,
    TOKENIZER_FILE,
    _default_prepared_source,
    _source_description,
    load_prepared_corpus,
)


def build_parser() -> argparse.ArgumentParser:
    parser = common.build_parser()
    parser.description = (
        "Train unique token attention with a shared SwiGLU FFN plus small "
        "token-specific low-rank FFN modifiers."
    )
    actions = {action.dest: action for action in parser._actions}
    actions["data"].required = False
    actions["d_model"].default = 16
    actions["heads"].default = 4
    actions["layers"].default = 1
    actions["ffn_dim"].default = 32
    actions["structure_interval"].default = 0
    actions["model"].choices = ("dt",)
    actions["model"].default = "dt"
    parser.add_argument(
        "--prepared-data",
        type=Path,
        default=None,
        help="Directory or ZIP containing tokenizer.json and prepared tokens.",
    )
    parser.add_argument("--ffn-mod-dim", type=int, default=4)
    parser.add_argument("--ffn-mod-scale", type=float, default=1.0)
    parser.add_argument("--max-parameters", type=int, default=300_000_000)
    parser.add_argument("--allow-large-model", action="store_true")
    return parser


def _build_config(args: argparse.Namespace, vocab_size: int) -> DynamicTransformerConfig:
    config = common._build_config(args, vocab_size)
    config.validate()
    return config


def _parameter_estimate(config: DynamicTransformerConfig, rank: int) -> int:
    v, d, f, l = config.vocab_size, config.d_model, config.ffn_dim, config.n_layers
    embedding_head = 2 * v * d
    norms = l * 2 * d + d
    unique_attention = l * v * 4 * d * d
    shared_ffn = l * 3 * d * f
    token_mod = l * v * 3 * rank * (d + f)
    return int(embedding_head + norms + unique_attention + shared_ffn + token_mod)


def _load_corpus(parser: argparse.ArgumentParser, args: argparse.Namespace):
    if args.data and args.prepared_data:
        parser.error("use either --data or --prepared-data, not both")
    prepared = args.prepared_data
    if not args.data and prepared is None:
        prepared = _default_prepared_source()
        if prepared is None:
            parser.error(
                f"provide --prepared-data or --data; expected "
                f"{DEFAULT_PREPARED_DIRECTORY} or {DEFAULT_PREPARED_ARCHIVE}"
            )
        args.prepared_data = prepared
    if prepared is not None:
        return (
            load_prepared_corpus(
                prepared,
                validation_fraction=args.validation_fraction,
                seq_len=args.seq_len,
            ),
            _source_description(prepared),
        )
    max_vocab = None if args.max_vocab == 0 else args.max_vocab
    corpus = common.prepare_corpus(
        args.data,
        lowercase=args.lowercase,
        min_frequency=args.min_frequency,
        max_vocab=max_vocab,
        validation_fraction=args.validation_fraction,
        seq_len=args.seq_len,
    )
    return corpus, {"mode": "raw_text", "files": [str(p) for p in args.data]}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    common._validate_args(args)
    if args.ffn_mod_dim <= 0:
        parser.error("--ffn-mod-dim must be positive")
    if args.weight_decay != 0:
        parser.error("this sparse lookup model requires --weight-decay 0")
    if args.structure_interval != 0:
        parser.error("this model has no split/merge; use --structure-interval 0")

    device = common._resolve_device(args.device)
    corpus, source = _load_corpus(parser, args)
    config = _build_config(args, corpus.tokenizer.vocab_size)
    estimate = _parameter_estimate(config, args.ffn_mod_dim)
    if estimate > args.max_parameters and not args.allow_large_model:
        parser.error(
            f"model estimate {estimate:,} exceeds --max-parameters; reduce rank, "
            "width, FFN, or depth, or pass --allow-large-model"
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

    common._set_seed(args.seed)
    model = UniqueAttentionSharedFFNMod(
        config,
        mod_rank=args.ffn_mod_dim,
        mod_scale=args.ffn_mod_scale,
    ).to(device)
    sparse = list(model.sparse_parameters())
    dense = list(model.dense_parameters())
    kwargs = {
        "lr": args.lr,
        "betas": (args.beta1, args.beta2),
        "eps": args.adam_eps,
    }
    optimizers = OptimizerSet(
        [
            torch.optim.SparseAdam(sparse, **kwargs),
            torch.optim.AdamW(dense, weight_decay=0.0, **kwargs),
        ]
    )

    snapshot = dict(vars(args))
    snapshot["data"] = [str(p) for p in args.data] if args.data else None
    snapshot["prepared_data"] = str(args.prepared_data) if args.prepared_data else None
    snapshot["output_dir"] = str(args.output_dir)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "arguments": snapshot,
                "model_config": asdict(config),
                "source": source,
                "parameter_estimate": estimate,
                "architecture": "unique attention + shared FFN + token low-rank MOD",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    def record(payload: dict) -> None:
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    initial_loss, initial_ppl = common.evaluate(
        model,
        corpus.validation_tokens,
        plan.validation_starts,
        seq_len=config.max_seq_len,
        device=device,
    )
    print(f"[unique_attn_shared_ffn_mod] step=0 val_loss={initial_loss:.4f} val_ppl={initial_ppl:.3f}")
    record({"step": 0, "validation_loss": initial_loss, "validation_ppl": initial_ppl})
    best_loss, best_step = initial_loss, 0
    processed, seconds, final_train = 0, 0.0, float("nan")
    optimizers.zero_grad()

    for step_index in range(args.steps):
        model.train()
        lr = common._learning_rate(
            step_index,
            total_steps=args.steps,
            base_lr=args.lr,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        optimizers.set_lr(lr)
        common._sync(device)
        started = time.perf_counter()
        losses = []
        for micro in range(args.grad_accum):
            batch = common.materialize_batch(
                corpus.train_tokens,
                plan.train_starts[step_index, micro],
                config.max_seq_len,
                device,
            )
            output = model(batch, labels=batch)
            assert output.loss is not None
            (output.loss / args.grad_accum).backward()
            losses.append(float(output.loss.detach().cpu()))
        sparse_norm = _clip_gradients(sparse, args.grad_clip)
        dense_norm = _clip_gradients(dense, args.grad_clip)
        optimizers.step()
        optimizers.zero_grad()
        common._sync(device)
        seconds += time.perf_counter() - started
        step = step_index + 1
        final_train = sum(losses) / len(losses)
        processed += args.batch_size * args.grad_accum * (config.max_seq_len - 1)
        if step % args.log_interval == 0 or step == 1:
            rate = processed / max(seconds, 1e-9)
            print(
                f"[unique_attn_shared_ffn_mod] step={step} train_loss={final_train:.4f} "
                f"lr={lr:.3e} tok/s={rate:.1f} sparse_grad={sparse_norm:.3f} "
                f"dense_grad={dense_norm:.3f}"
            )
            record({"step": step, "train_loss": final_train, "learning_rate": lr,
                    "tokens_per_second": rate, "sparse_gradient_norm": sparse_norm,
                    "dense_gradient_norm": dense_norm})
        if step % args.eval_interval == 0 or step == args.steps:
            val_loss, val_ppl = common.evaluate(
                model,
                corpus.validation_tokens,
                plan.validation_starts,
                seq_len=config.max_seq_len,
                device=device,
            )
            if val_loss < best_loss:
                best_loss, best_step = val_loss, step
            print(f"[unique_attn_shared_ffn_mod] step={step} val_loss={val_loss:.4f} val_ppl={val_ppl:.3f}")
            record({"step": step, "validation_loss": val_loss, "validation_ppl": val_ppl})

    final_loss, final_ppl = common.evaluate(
        model,
        corpus.validation_tokens,
        plan.validation_starts,
        seq_len=config.max_seq_len,
        device=device,
    )
    summary = {
        "model": "unique_attn_shared_ffn_mod",
        "mod_rank": args.ffn_mod_dim,
        "mod_scale": args.ffn_mod_scale,
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "best_validation_ppl": math.exp(min(best_loss, 30.0)),
        "final_train_loss": final_train,
        "final_validation_loss": final_loss,
        "final_validation_ppl": final_ppl,
        "processed_target_tokens": processed,
        "tokens_per_second": processed / max(seconds, 1e-9),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        **model.lookup_summary(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not args.no_save_checkpoint:
        torch.save(
            {"model": model.state_dict(), "config": asdict(config), "summary": summary},
            run_dir / "checkpoint.pt",
        )
    print("\nFinal FFN MOD result")
    print(
        f"val_ppl={final_ppl:.3f} best_ppl={summary['best_validation_ppl']:.3f} "
        f"best_step={best_step:,} parameters={summary['total_parameters']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
