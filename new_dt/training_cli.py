from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from . import comparison as base
from .word_tokenizer import WordSpaceTokenizer


TOKEN_FILE = "pretrain_train_tokens.pt"
PACKED_TOKEN_FILE = "pretrain_train_tokens.pt.gz.b64"
TOKENIZER_FILE = "tokenizer.json"
METADATA_FILE = "metadata.json"


def _split_stream(
    token_ids: Tensor,
    *,
    validation_fraction: float,
    seq_len: int,
) -> tuple[Tensor, Tensor]:
    if token_ids.ndim != 1:
        raise ValueError("prepared input_ids must be a one-dimensional token stream")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")

    required = 2 * (seq_len + 1)
    if token_ids.numel() < required:
        raise ValueError(
            "prepared token stream is too small for the requested sequence length: "
            f"tokens={token_ids.numel()}, seq_len={seq_len}; need at least {required}"
        )

    validation_count = max(
        seq_len + 1,
        int(round(token_ids.numel() * validation_fraction)),
    )
    validation_count = min(
        validation_count,
        token_ids.numel() - (seq_len + 1),
    )
    split = token_ids.numel() - validation_count
    return token_ids[:split].clone(), token_ids[split:].clone()


def _load_token_payload(directory: Path) -> tuple[dict[str, Any], str]:
    direct_path = directory / TOKEN_FILE
    if direct_path.is_file():
        payload = torch.load(direct_path, map_location="cpu", weights_only=False)
        return payload, direct_path.name

    packed_path = directory / PACKED_TOKEN_FILE
    if packed_path.is_file():
        encoded = "".join(packed_path.read_text(encoding="ascii").split())
        try:
            compressed = base64.b64decode(encoded, validate=True)
            raw = gzip.decompress(compressed)
        except Exception as error:
            raise ValueError(f"invalid prepared token archive: {packed_path}") from error
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        return payload, packed_path.name

    raise FileNotFoundError(
        f"{directory} must contain {TOKEN_FILE} or {PACKED_TOKEN_FILE}"
    )


def load_prepared_corpus(
    directory: str | Path,
    *,
    validation_fraction: float,
    seq_len: int,
) -> base.CorpusBundle:
    """Load the exact saved tokenizer and pre-tokenized stream without rebuilding IDs."""

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(root)

    tokenizer_path = root / TOKENIZER_FILE
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    tokenizer = WordSpaceTokenizer.load(tokenizer_path)

    payload, payload_file = _load_token_payload(root)
    if not isinstance(payload, dict):
        raise ValueError("prepared token payload must be a dictionary")
    if payload.get("format") != "new-dt-word-space-token-stream":
        raise ValueError("unsupported prepared token stream format")
    if int(payload.get("version", 0)) != 1:
        raise ValueError("unsupported prepared token stream version")

    input_ids = payload.get("input_ids")
    if not isinstance(input_ids, Tensor):
        raise ValueError("prepared token payload has no input_ids tensor")
    if input_ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("prepared input_ids must use an integer dtype")
    input_ids = input_ids.to(dtype=torch.long, device="cpu").contiguous()

    expected_vocab = int(payload.get("vocab_size", -1))
    if expected_vocab != tokenizer.vocab_size:
        raise ValueError(
            "tokenizer/payload vocabulary mismatch: "
            f"tokenizer={tokenizer.vocab_size}, payload={expected_vocab}"
        )
    if int(payload.get("eos_id", -1)) != tokenizer.eos_id:
        raise ValueError("tokenizer/payload EOS ID mismatch")
    if input_ids.numel() and (
        int(input_ids.min()) < 0 or int(input_ids.max()) >= tokenizer.vocab_size
    ):
        raise ValueError("prepared token IDs fall outside the tokenizer vocabulary")

    metadata_path = root / METADATA_FILE
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_vocab = int(metadata.get("tokenizer", {}).get("vocab_size", -1))
        if metadata_vocab != tokenizer.vocab_size:
            raise ValueError("metadata/tokenizer vocabulary mismatch")
        metadata_tokens = int(
            metadata.get("pretraining_corpus", {}).get("encoded_tokens", -1)
        )
        if metadata_tokens != input_ids.numel():
            raise ValueError("metadata/token-stream length mismatch")

    train, validation = _split_stream(
        input_ids,
        validation_fraction=validation_fraction,
        seq_len=seq_len,
    )
    source_characters = int(
        metadata.get("pretraining_corpus", {}).get("characters", 0)
    )
    corpus = base.CorpusBundle(tokenizer, train, validation, source_characters)
    setattr(corpus, "prepared_payload_file", payload_file) if hasattr(
        corpus, "__dict__"
    ) else None
    return corpus


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    data_action = next(action for action in parser._actions if action.dest == "data")
    data_action.required = False
    data_action.help = "Raw UTF-8 text files; tokenizer IDs are rebuilt from these files."
    parser.add_argument(
        "--prepared-data",
        type=Path,
        default=None,
        help=(
            "Directory containing tokenizer.json and a prepared token stream. "
            "When neither source option is supplied, data/sciq is used if present."
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
        default = Path("data/sciq")
        if (default / TOKENIZER_FILE).is_file():
            prepared = default
            args.prepared_data = default
        else:
            parser.error("provide --prepared-data or --data")

    if prepared is not None:
        corpus = load_prepared_corpus(
            prepared,
            validation_fraction=args.validation_fraction,
            seq_len=args.seq_len,
        )
        return corpus, {
            "mode": "prepared",
            "directory": str(prepared),
            "tokenizer": str(prepared / TOKENIZER_FILE),
            "token_stream": str(
                prepared
                / (
                    TOKEN_FILE
                    if (prepared / TOKEN_FILE).is_file()
                    else PACKED_TOKEN_FILE
                )
            ),
        }

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
        "fairness": {
            "same_tokenizer": True,
            "same_precomputed_token_stream": True,
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

    names: list[base.ModelName]
    if args.model == "both":
        names = ["gpt", "sdt"]
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
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    base._print_comparison(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
