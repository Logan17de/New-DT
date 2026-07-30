from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from .config import DynamicTransformerConfig
from .small_gpt import SmallGPT
from .word_tokenizer import WordSpaceTokenizer

ModelName = Literal["gpt", "sdt"]


@dataclass(slots=True)
class CorpusBundle:
    tokenizer: WordSpaceTokenizer
    train_tokens: Tensor
    validation_tokens: Tensor
    source_characters: int


@dataclass(slots=True)
class BatchPlan:
    """Precomputed starts reused unchanged by both comparison models."""

    train_starts: Tensor
    validation_starts: Tensor


def _read_corpus(paths: list[Path]) -> str:
    if not paths:
        raise ValueError("at least one --data file is required")
    chunks: list[str] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def prepare_corpus(
    paths: list[Path],
    *,
    lowercase: bool,
    min_frequency: int,
    max_vocab: int | None,
    validation_fraction: float,
    seq_len: int,
) -> CorpusBundle:
    text = _read_corpus(paths)
    tokenizer = WordSpaceTokenizer.train(
        text,
        lowercase=lowercase,
        min_frequency=min_frequency,
        max_vocab=max_vocab,
    )
    token_ids = tokenizer.encode_document(text)
    required = 2 * (seq_len + 1)
    if len(token_ids) < required:
        raise ValueError(
            "dataset is too small for the requested sequence length: "
            f"tokens={len(token_ids)}, seq_len={seq_len}; provide at least "
            f"{required} tokens"
        )
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")

    validation_count = max(
        seq_len + 1, int(round(len(token_ids) * validation_fraction))
    )
    validation_count = min(validation_count, len(token_ids) - (seq_len + 1))
    split = len(token_ids) - validation_count
    train = torch.tensor(token_ids[:split], dtype=torch.long)
    validation = torch.tensor(token_ids[split:], dtype=torch.long)
    return CorpusBundle(tokenizer, train, validation, len(text))


def _random_starts(
    token_count: int,
    *,
    shape: tuple[int, ...],
    seq_len: int,
    generator: torch.Generator,
) -> Tensor:
    max_start = token_count - seq_len
    if max_start < 0:
        raise ValueError(
            f"token stream ({token_count}) is shorter than seq_len ({seq_len})"
        )
    return torch.randint(
        0,
        max_start + 1,
        shape,
        generator=generator,
        dtype=torch.long,
    )


def make_batch_plan(
    corpus: CorpusBundle,
    *,
    steps: int,
    grad_accum: int,
    batch_size: int,
    eval_batches: int,
    seq_len: int,
    seed: int,
) -> BatchPlan:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    train_starts = _random_starts(
        len(corpus.train_tokens),
        shape=(steps, grad_accum, batch_size),
        seq_len=seq_len,
        generator=generator,
    )
    validation_starts = _random_starts(
        len(corpus.validation_tokens),
        shape=(eval_batches, batch_size),
        seq_len=seq_len,
        generator=generator,
    )
    return BatchPlan(train_starts, validation_starts)


def materialize_batch(
    stream: Tensor,
    starts: Tensor,
    seq_len: int,
    device: torch.device,
) -> Tensor:
    offsets = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    batch = stream[starts.reshape(-1, 1) + offsets]
    return batch.to(device=device, non_blocking=True)


def build_model(model_name: ModelName, config: DynamicTransformerConfig) -> nn.Module:
    if model_name == "gpt":
        return SmallGPT(config)
    if model_name == "sdt":
        from .model import DynamicTransformer

        return DynamicTransformer(config)
    raise ValueError(f"unknown model: {model_name}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _learning_rate(
    step_index: int,
    *,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if warmup_steps > 0 and step_index < warmup_steps:
        return base_lr * float(step_index + 1) / float(warmup_steps)
    if total_steps <= warmup_steps + 1:
        return base_lr
    progress = (step_index - warmup_steps) / max(
        1, total_steps - warmup_steps - 1
    )
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def evaluate(
    model: nn.Module,
    stream: Tensor,
    starts: Tensor,
    *,
    seq_len: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_starts in starts:
            batch = materialize_batch(stream, batch_starts, seq_len, device)
            output = model(batch, labels=batch)
            if output.loss is None:
                raise RuntimeError("model did not return an evaluation loss")
            losses.append(float(output.loss.detach().cpu()))
    mean_loss = sum(losses) / max(1, len(losses))
    return mean_loss, math.exp(min(mean_loss, 30.0))


def _parameter_stats(model: nn.Module) -> dict[str, int]:
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    result = {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "parameter_bytes": int(parameter_bytes),
    }
    if hasattr(model, "pool_summary"):
        pools = model.pool_summary()  # type: ignore[no-untyped-call]
        pool_capacity = sum(int(item["capacity"]) for item in pools.values())
        active_scalars = sum(int(item["active"]) for item in pools.values())
        result.update(
            {
                "pool_capacity": pool_capacity,
                "active_scalars": active_scalars,
                "effective_active_parameters": total
                - pool_capacity
                + active_scalars,
            }
        )
    return result


def _route_stats(model: nn.Module) -> dict[str, int]:
    if not hasattr(model, "routing_storage_summary"):
        return {"route_storage_bytes": 0, "logical_route_references": 0}
    storage = model.routing_storage_summary()  # type: ignore[no-untyped-call]
    logical = 0
    for _, routed in model.routed_tensors():  # type: ignore[no-untyped-call]
        logical += int(routed.vocab_size) * int(routed.route_size)
    return {
        "route_storage_bytes": int(storage["total_bytes"]),
        "logical_route_references": logical,
    }


def _assert_untied_head(model_name: ModelName, model: nn.Module) -> None:
    if model_name == "gpt":
        if not bool(model.lm_is_untied):  # type: ignore[attr-defined]
            raise RuntimeError("GPT embedding and LM head unexpectedly share storage")
        return
    embedding_pool = model.embedding.parameters_by_token.pool.values  # type: ignore[attr-defined]
    output_pool = model.lm_head.parameters_by_token.pool.values  # type: ignore[attr-defined]
    if embedding_pool is output_pool or embedding_pool.data_ptr() == output_pool.data_ptr():
        raise RuntimeError("sDT embedding and LM head unexpectedly share storage")


def _make_controller(model_name: ModelName, args: argparse.Namespace):
    if model_name != "sdt" or args.structure_interval <= 0:
        return None
    if args.weight_decay != 0.0 and not args.no_merge:
        raise ValueError(
            "sDT merge indexing requires --weight-decay 0; use --no-merge "
            "when applying weight decay"
        )
    from .structure import DynamicStructureController

    return DynamicStructureController(
        structure_interval=args.structure_interval,
        ema_decay=args.structure_ema_decay,
        min_owner_samples=args.min_owner_samples,
        min_gradient_magnitude=args.min_gradient_magnitude,
        min_conflict_score=args.min_conflict_score,
        owner_threshold_scale=args.owner_threshold_scale,
        max_conflict_threshold=args.max_conflict_threshold,
        max_splits_per_pass=args.max_splits_per_pass,
        enable_merge=not args.no_merge,
        merge_weight_tolerance=args.merge_weight_tolerance,
        merge_gradient_tolerance=args.merge_gradient_tolerance,
        merge_min_samples=args.merge_min_samples,
        max_merges_per_pass=args.max_merges_per_pass,
    )


def _generate(
    model: nn.Module,
    tokenizer: WordSpaceTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    max_seq_len: int,
) -> str:
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_id]
    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated[:, -max_seq_len:]
            logits = model(context).logits[:, -1]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            if int(next_token.item()) == tokenizer.eos_id:
                break
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


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
    # Reset before each construction so both experiments begin from the same RNG
    # state and consume the exact same precomputed token batches.
    _set_seed(args.seed)
    model = build_model(model_name, config).to(device)
    _assert_untied_head(model_name, model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    controller = _make_controller(model_name, args)
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
    split_events = 0
    merge_events = 0
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

    optimizer.zero_grad(set_to_none=True)
    for step_index in range(args.steps):
        model.train()
        lr = _learning_rate(
            step_index,
            total_steps=args.steps,
            base_lr=args.lr,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        _sync(device)
        started = time.perf_counter()
        micro_losses: list[float] = []
        for micro_index in range(args.grad_accum):
            starts = plan.train_starts[step_index, micro_index]
            batch = materialize_batch(
                corpus.train_tokens, starts, config.max_seq_len, device
            )
            output = model(
                batch,
                labels=batch,
                collect_route_grads=controller is not None,
            )
            if output.loss is None:
                raise RuntimeError("model did not return a training loss")
            (output.loss / args.grad_accum).backward()
            micro_losses.append(float(output.loss.detach().cpu()))
            if controller is not None:
                controller.collect(model)

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer_step = step_index + 1
        events = []
        if controller is not None:
            events = controller.maybe_restructure(
                model,
                optimizer,
                optimizer_step=optimizer_step,
            )
            split_events += sum(event.kind == "split" for event in events)
            merge_events += sum(event.kind == "merge" for event in events)
        optimizer.zero_grad(set_to_none=True)
        _sync(device)
        train_seconds += time.perf_counter() - started

        final_train_loss = sum(micro_losses) / len(micro_losses)
        processed_tokens += (
            args.batch_size * args.grad_accum * (config.max_seq_len - 1)
        )
        if optimizer_step % args.log_interval == 0 or optimizer_step == 1:
            tokens_per_second = processed_tokens / max(train_seconds, 1e-9)
            print(
                f"[{model_name}] step={optimizer_step} "
                f"train_loss={final_train_loss:.4f} lr={lr:.3e} "
                f"tok/s={tokens_per_second:.1f} events={len(events)}"
            )
            record(
                {
                    "step": optimizer_step,
                    "train_loss": final_train_loss,
                    "learning_rate": lr,
                    "tokens_per_second": tokens_per_second,
                    "structure_events": len(events),
                }
            )

        should_eval = (
            optimizer_step % args.eval_interval == 0
            or optimizer_step == args.steps
        )
        if should_eval:
            val_loss, val_ppl = evaluate(
                model,
                corpus.validation_tokens,
                plan.validation_starts,
                seq_len=config.max_seq_len,
                device=device,
            )
            best_loss = min(best_loss, val_loss)
            print(
                f"[{model_name}] step={optimizer_step} "
                f"val_loss={val_loss:.4f} val_ppl={val_ppl:.3f}"
            )
            record(
                {
                    "step": optimizer_step,
                    "validation_loss": val_loss,
                    "validation_ppl": val_ppl,
                }
            )

    if controller is not None:
        final_events = controller.maybe_restructure(
            model,
            optimizer,
            optimizer_step=args.steps,
            force=True,
        )
        split_events += sum(event.kind == "split" for event in final_events)
        merge_events += sum(event.kind == "merge" for event in final_events)

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
        "split_events": split_events,
        "merge_events": merge_events,
        "sample": sample,
        **_parameter_stats(model),
        **_route_stats(model),
    }
    (model_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
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
    if model_name == "sdt" and args.export_routing:
        model.export_routing(model_dir / "routing")  # type: ignore[no-untyped-call]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _build_config(
    args: argparse.Namespace, vocab_size: int
) -> DynamicTransformerConfig:
    ffn_dim = args.ffn_dim if args.ffn_dim is not None else 4 * args.d_model
    return DynamicTransformerConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        ffn_dim=ffn_dim,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        initial_shared_fraction=args.initial_shared_fraction,
        pool_growth_factor=args.pool_growth_factor,
        init_std=args.init_std,
        route_page_size=args.route_page_size,
        route_templates_per_page=args.route_templates_per_page,
        route_delta_promotion_threshold=args.route_delta_promotion_threshold,
        route_template_promotion_threshold=args.route_template_promotion_threshold,
        route_template_promotion_fraction=args.route_template_promotion_fraction,
        route_shared_delta_min_reuse=args.route_shared_delta_min_reuse,
        route_cache_pages=args.route_cache_pages,
        route_linear_out_tile=args.route_linear_out_tile,
        route_lm_head_tile=args.route_lm_head_tile,
        route_materialize_token_chunk=args.route_materialize_token_chunk,
        rope_theta=args.rope_theta,
    )


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _print_comparison(results: list[dict[str, Any]]) -> None:
    print("\nFinal comparison")
    header = (
        f"{'model':<8} {'val ppl':>10} {'best ppl':>10} {'tok/s':>12} "
        f"{'trainable':>12} {'effective':>12} {'route MB':>10}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        effective = item.get(
            "effective_active_parameters", item["trainable_parameters"]
        )
        print(
            f"{item['model']:<8} {item['final_validation_ppl']:>10.3f} "
            f"{item['best_validation_ppl']:>10.3f} "
            f"{item['tokens_per_second']:>12.1f} "
            f"{item['trainable_parameters']:>12,d} {effective:>12,d} "
            f"{item['route_storage_bytes'] / 1_000_000:>10.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an untied small shared GPT and sDT on exactly the same "
            "word-token batches."
        )
    )
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--model", choices=("gpt", "sdt", "both"), default="both"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/small_comparison")
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)

    tokenizer = parser.add_argument_group("word-space tokenizer")
    tokenizer.add_argument("--lowercase", action="store_true")
    tokenizer.add_argument("--min-frequency", type=int, default=1)
    tokenizer.add_argument(
        "--max-vocab", type=int, default=0, help="0 keeps every word"
    )
    tokenizer.add_argument("--validation-fraction", type=float, default=0.1)

    architecture = parser.add_argument_group("shared architecture")
    architecture.add_argument("--d-model", type=int, default=32)
    architecture.add_argument("--heads", type=int, default=4)
    architecture.add_argument("--layers", type=int, default=2)
    architecture.add_argument("--ffn-dim", type=int, default=None)
    architecture.add_argument("--seq-len", type=int, default=64)
    architecture.add_argument("--dropout", type=float, default=0.0)
    architecture.add_argument("--init-std", type=float, default=0.02)
    architecture.add_argument("--rope-theta", type=float, default=10_000.0)

    training = parser.add_argument_group("training")
    training.add_argument("--steps", type=int, default=500)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--grad-accum", type=int, default=1)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--warmup-steps", type=int, default=20)
    training.add_argument("--min-lr-ratio", type=float, default=0.1)
    training.add_argument("--beta1", type=float, default=0.9)
    training.add_argument("--beta2", type=float, default=0.95)
    training.add_argument("--adam-eps", type=float, default=1e-8)
    training.add_argument("--weight-decay", type=float, default=0.0)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument("--eval-interval", type=int, default=50)
    training.add_argument("--eval-batches", type=int, default=20)
    training.add_argument("--log-interval", type=int, default=10)
    training.add_argument("--no-save-checkpoint", action="store_true")
    training.add_argument("--sample-prompt", default=None)
    training.add_argument("--sample-tokens", type=int, default=40)
    training.add_argument("--temperature", type=float, default=0.0)

    routes = parser.add_argument_group("sDT route storage")
    routes.add_argument("--initial-shared-fraction", type=float, default=0.5)
    routes.add_argument("--pool-growth-factor", type=float, default=1.5)
    routes.add_argument("--route-page-size", type=int, default=256)
    routes.add_argument("--route-templates-per-page", type=int, default=4)
    routes.add_argument(
        "--route-delta-promotion-threshold", type=int, default=16
    )
    routes.add_argument(
        "--route-template-promotion-threshold", type=int, default=64
    )
    routes.add_argument(
        "--route-template-promotion-fraction", type=float, default=0.25
    )
    routes.add_argument("--route-shared-delta-min-reuse", type=int, default=2)
    routes.add_argument("--route-cache-pages", type=int, default=256)
    routes.add_argument("--route-linear-out-tile", type=int, default=32)
    routes.add_argument("--route-lm-head-tile", type=int, default=256)
    routes.add_argument("--route-materialize-token-chunk", type=int, default=128)
    routes.add_argument("--export-routing", action="store_true")

    structure = parser.add_argument_group(
        "sDT split/merge; set interval 0 to disable"
    )
    structure.add_argument("--structure-interval", type=int, default=100)
    structure.add_argument("--structure-ema-decay", type=float, default=0.95)
    structure.add_argument("--min-owner-samples", type=int, default=8)
    structure.add_argument("--min-gradient-magnitude", type=float, default=1e-5)
    structure.add_argument("--min-conflict-score", type=float, default=0.6)
    structure.add_argument("--owner-threshold-scale", type=float, default=0.03)
    structure.add_argument("--max-conflict-threshold", type=float, default=0.95)
    structure.add_argument("--max-splits-per-pass", type=int, default=8)
    structure.add_argument("--no-merge", action="store_true")
    structure.add_argument("--merge-weight-tolerance", type=float, default=1e-5)
    structure.add_argument(
        "--merge-gradient-tolerance", type=float, default=1e-5
    )
    structure.add_argument("--merge-min-samples", type=int, default=1)
    structure.add_argument("--max-merges-per-pass", type=int, default=8)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "eval_interval": args.eval_interval,
        "eval_batches": args.eval_batches,
        "log_interval": args.log_interval,
        "seq_len": args.seq_len,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.seq_len < 2:
        raise ValueError("--seq-len must be at least 2")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps cannot be negative")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    device = _resolve_device(args.device)
    max_vocab = None if args.max_vocab == 0 else args.max_vocab
    corpus = prepare_corpus(
        args.data,
        lowercase=args.lowercase,
        min_frequency=args.min_frequency,
        max_vocab=max_vocab,
        validation_fraction=args.validation_fraction,
        seq_len=args.seq_len,
    )
    config = _build_config(args, corpus.tokenizer.vocab_size)
    config.validate()
    plan = make_batch_plan(
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
    corpus.tokenizer.save(run_dir / "tokenizer.json")
    argument_snapshot = dict(vars(args))
    argument_snapshot["data"] = [str(path) for path in args.data]
    argument_snapshot["output_dir"] = str(args.output_dir)
    run_config = {
        "arguments": argument_snapshot,
        "model_config": asdict(config),
        "dataset": {
            "source_characters": corpus.source_characters,
            "vocab_size": corpus.tokenizer.vocab_size,
            "train_tokens": len(corpus.train_tokens),
            "validation_tokens": len(corpus.validation_tokens),
            "target_tokens_per_step": args.batch_size
            * args.grad_accum
            * (args.seq_len - 1),
        },
        "fairness": {
            "same_tokenizer": True,
            "same_train_start_plan": True,
            "same_validation_start_plan": True,
            "same_optimizer_hyperparameters": True,
            "same_rope_rmsnorm_swiglu_topology": True,
            "untied_lm_heads": True,
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )

    names: list[ModelName]
    if args.model == "both":
        names = ["gpt", "sdt"]
    else:
        names = [args.model]
    print(
        f"run={run_dir} device={device} vocab={config.vocab_size} "
        f"train_tokens={len(corpus.train_tokens)} "
        f"val_tokens={len(corpus.validation_tokens)}"
    )
    results = [
        train_model(
            name,
            config,
            corpus,
            plan,
            args=args,
            run_dir=run_dir,
            device=device,
        )
        for name in names
    ]
    (run_dir / "comparison.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    _print_comparison(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
