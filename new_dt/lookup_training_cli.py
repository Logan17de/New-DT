from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from . import lookup_comparison as base
from .training_cli import (
    DEFAULT_PREPARED_ARCHIVE,
    DEFAULT_PREPARED_DIRECTORY,
    TOKENIZER_FILE,
    _default_prepared_source,
    _source_description,
    load_prepared_corpus,
)


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    data_action = next(action for action in parser._actions if action.dest == "data")
    data_action.required = False
    data_action.help = "Raw UTF-8 text files; a tokenizer is trained from these files."
    parser.add_argument(
        "--prepared-data",
        type=Path,
        default=None,
        help=(
            "Directory or ZIP containing tokenizer.json and the prepared token stream. "
            "When omitted, data/sciq or data/sciq.zip is selected automatically."
        ),
    )
    return parser


def _resolve_corpus_source(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[base.CorpusBundle, dict[str, Any]]:
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
    corpus = base.prepare_corpus(
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    base._validate_args(args)
    device = base._resolve_device(args.device)
    corpus, source = _resolve_corpus_source(parser, args)

    config = base._build_config(args, corpus.tokenizer.vocab_size)
    config.validate()
    parameter_counts = base.estimate_parameter_counts(config)
    training_bytes = base.estimate_training_bytes(config)

    print(
        "model estimates: "
        f"GPT={parameter_counts['gpt']:,} params "
        f"(~{training_bytes['gpt'] / 1024**3:.2f} GiB FP32 training state), "
        f"DT={parameter_counts['dt']:,} params "
        f"(~{training_bytes['dt'] / 1024**3:.2f} GiB FP32 training state)"
    )
    print(
        "DT mode: direct per-token lookup tables for embedding, Q/K/V/O, "
        "SwiGLU Up/Gate/Down, and an untied output-token table; "
        "split/merge and SPRC are disabled."
    )

    plan = base.make_batch_plan(
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
        str(args.prepared_data) if args.prepared_data is not None else None
    )
    argument_snapshot["output_dir"] = str(args.output_dir)
    run_config = {
        "arguments": argument_snapshot,
        "model_config": asdict(config),
        "parameter_estimates": parameter_counts,
        "training_state_byte_estimates": training_bytes,
        "dataset": {
            "source": source,
            "source_characters": corpus.source_characters,
            "vocab_size": corpus.tokenizer.vocab_size,
            "train_tokens": len(corpus.train_tokens),
            "validation_tokens": len(corpus.validation_tokens),
            "target_tokens_per_step": args.batch_size
            * args.grad_accum
            * (args.seq_len - 1),
        },
        "comparison": {
            "models": ["gpt", "dt"],
            "gpt": "conventional shared Q/K/V/O and FFN matrices",
            "dt": "independent per-token embedding, attention, and FFN lookup rows",
            "structural_updates": False,
            "route_compression": False,
            "untied_lm_heads": True,
        },
        "fairness": {
            "same_saved_tokenizer": True,
            "same_precomputed_token_stream": True,
            "same_train_start_plan": True,
            "same_validation_start_plan": True,
            "same_depth_width_rope_rmsnorm_swiglu_topology": True,
            "same_learning_rate_schedule_and_adam_hyperparameters": True,
            "optimizer_note": (
                "GPT uses dense AdamW. DT uses SparseAdam for active token lookup "
                "rows and AdamW for dense normalization/output parameters."
            ),
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    names: list[base.ModelName]
    if args.model == "both":
        names = ["gpt", "dt"]
    else:
        names = [args.model]

    print(
        f"run={run_dir} device={device} vocab={config.vocab_size} "
        f"train_tokens={len(corpus.train_tokens)} "
        f"val_tokens={len(corpus.validation_tokens)} source={source['mode']}"
    )
    results = [
        base.train_model(
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
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    base._print_comparison(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
