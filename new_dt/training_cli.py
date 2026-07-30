from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from torch import Tensor

from . import comparison as base
from .word_tokenizer import WordSpaceTokenizer


TOKEN_FILE = "pretrain_train_tokens.pt"
PACKED_TOKEN_FILE = "pretrain_train_tokens.pt.gz.b64"
TOKENIZER_FILE = "tokenizer.json"
METADATA_FILE = "metadata.json"
DEFAULT_PREPARED_DIRECTORY = Path("data/sciq")
DEFAULT_PREPARED_ARCHIVE = Path("data/sciq.zip")


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


def _tokenizer_from_json(raw: str, *, source: str) -> WordSpaceTokenizer:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid tokenizer JSON: {source}") from error
    if payload.get("type") != "word_space":
        raise ValueError(f"not a WordSpaceTokenizer file: {source}")
    if int(payload.get("version", 1)) != 1:
        raise ValueError(f"unsupported tokenizer version: {source}")
    return WordSpaceTokenizer(
        tokens=list(payload["tokens"]),
        lowercase=bool(payload.get("lowercase", False)),
    )


def _decode_packed_payload(encoded_text: str, *, source: str) -> dict[str, Any]:
    encoded = "".join(encoded_text.split())
    try:
        compressed = base64.b64decode(encoded, validate=True)
        raw = gzip.decompress(compressed)
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"invalid prepared token archive: {source}") from error
    return payload


def _load_token_payload(directory: Path) -> tuple[dict[str, Any], str]:
    direct_path = directory / TOKEN_FILE
    if direct_path.is_file():
        payload = torch.load(direct_path, map_location="cpu", weights_only=False)
        return payload, direct_path.name

    packed_path = directory / PACKED_TOKEN_FILE
    if packed_path.is_file():
        payload = _decode_packed_payload(
            packed_path.read_text(encoding="ascii"),
            source=str(packed_path),
        )
        return payload, packed_path.name

    raise FileNotFoundError(
        f"{directory} must contain {TOKEN_FILE} or {PACKED_TOKEN_FILE}"
    )


def _find_archive_member(archive: zipfile.ZipFile, filename: str) -> str | None:
    matches = [
        name
        for name in archive.namelist()
        if not name.endswith("/") and PurePosixPath(name).name == filename
    ]
    if not matches:
        return None
    matches.sort(key=lambda name: (len(PurePosixPath(name).parts), name))
    return matches[0]


def _load_archive_components(
    archive_path: Path,
) -> tuple[WordSpaceTokenizer, dict[str, Any], dict[str, Any], str]:
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"prepared data file is not a ZIP archive: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        tokenizer_member = _find_archive_member(archive, TOKENIZER_FILE)
        if tokenizer_member is None:
            raise FileNotFoundError(
                f"{archive_path} does not contain {TOKENIZER_FILE}"
            )
        tokenizer = _tokenizer_from_json(
            archive.read(tokenizer_member).decode("utf-8"),
            source=f"{archive_path}!/{tokenizer_member}",
        )

        direct_member = _find_archive_member(archive, TOKEN_FILE)
        packed_member = _find_archive_member(archive, PACKED_TOKEN_FILE)
        if direct_member is not None:
            payload = torch.load(
                io.BytesIO(archive.read(direct_member)),
                map_location="cpu",
                weights_only=False,
            )
            payload_file = direct_member
        elif packed_member is not None:
            payload = _decode_packed_payload(
                archive.read(packed_member).decode("ascii"),
                source=f"{archive_path}!/{packed_member}",
            )
            payload_file = packed_member
        else:
            raise FileNotFoundError(
                f"{archive_path} must contain {TOKEN_FILE} or {PACKED_TOKEN_FILE}"
            )

        metadata: dict[str, Any] = {}
        metadata_member = _find_archive_member(archive, METADATA_FILE)
        if metadata_member is not None:
            metadata = json.loads(archive.read(metadata_member).decode("utf-8"))

    return tokenizer, payload, metadata, payload_file


def _validate_and_build_corpus(
    tokenizer: WordSpaceTokenizer,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    *,
    validation_fraction: float,
    seq_len: int,
) -> base.CorpusBundle:
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

    if metadata:
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
    return base.CorpusBundle(tokenizer, train, validation, source_characters)


def load_prepared_corpus(
    source: str | Path,
    *,
    validation_fraction: float,
    seq_len: int,
) -> base.CorpusBundle:
    """Load saved tokenizer IDs from a directory or ZIP without retokenizing."""

    root = Path(source)
    if root.is_file():
        tokenizer, payload, metadata, _ = _load_archive_components(root)
        return _validate_and_build_corpus(
            tokenizer,
            payload,
            metadata,
            validation_fraction=validation_fraction,
            seq_len=seq_len,
        )

    if not root.is_dir():
        raise FileNotFoundError(root)

    tokenizer_path = root / TOKENIZER_FILE
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    tokenizer = WordSpaceTokenizer.load(tokenizer_path)
    payload, _ = _load_token_payload(root)

    metadata_path = root / METADATA_FILE
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return _validate_and_build_corpus(
        tokenizer,
        payload,
        metadata,
        validation_fraction=validation_fraction,
        seq_len=seq_len,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    data_action = next(action for action in parser._actions if action.dest == "data")
    data_action.required = False
    data_action.help = "Raw UTF-8 text files; tokenizer IDs are rebuilt from these files."

    model_action = next(action for action in parser._actions if action.dest == "model")
    model_action.choices = ("gpt", "dt", "both")
    model_action.help = "Train GPT, DT, or the primary two-model comparison."

    parser.add_argument(
        "--prepared-data",
        type=Path,
        default=None,
        help=(
            "Directory or ZIP containing tokenizer.json and a prepared token stream. "
            "When omitted, data/sciq or data/sciq.zip is selected automatically."
        ),
    )
    return parser


def _default_prepared_source() -> Path | None:
    if (DEFAULT_PREPARED_DIRECTORY / TOKENIZER_FILE).is_file():
        return DEFAULT_PREPARED_DIRECTORY
    if DEFAULT_PREPARED_ARCHIVE.is_file():
        return DEFAULT_PREPARED_ARCHIVE
    return None


def _source_description(prepared: Path) -> dict[str, Any]:
    if prepared.is_file():
        return {
            "mode": "prepared_zip",
            "archive": str(prepared),
        }
    token_file = TOKEN_FILE if (prepared / TOKEN_FILE).is_file() else PACKED_TOKEN_FILE
    return {
        "mode": "prepared_directory",
        "directory": str(prepared),
        "tokenizer": str(prepared / TOKENIZER_FILE),
        "token_stream": str(prepared / token_file),
    }


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
                "provide --prepared-data or --data; expected data/sciq or data/sciq.zip"
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
            "primary_models": ["gpt", "dt"],
            "same_tokenizer": True,
            "same_precomputed_token_stream": True,
            "same_train_start_plan": True,
            "same_validation_start_plan": True,
            "same_optimizer_hyperparameters": True,
            "same_rope_rmsnorm_swiglu_topology": True,
            "untied_lm_heads": True,
        },
        "dt_structure": {
            "enabled": args.structure_interval > 0,
            "interval": args.structure_interval,
            "note": (
                "Normal DT split/merge is enabled."
                if args.structure_interval > 0
                else "Static-routing diagnostic ablation; not a separate primary model."
            ),
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )

    names: list[base.ModelName]
    if args.model == "both":
        names = ["gpt", "sdt"]
    elif args.model == "dt":
        names = ["sdt"]
    else:
        names = ["gpt"]

    print(
        f"run={run_dir} device={device} vocab={config.vocab_size} "
        f"train_tokens={len(corpus.train_tokens)} "
        f"val_tokens={len(corpus.validation_tokens)} source={source['mode']}"
    )
    print(
        "primary comparison: GPT vs DT; "
        f"DT split/merge={'enabled' if args.structure_interval > 0 else 'disabled'}"
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
    for result in results:
        if result.get("model") == "sdt":
            result["model"] = "dt"

    (run_dir / "comparison.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    base._print_comparison(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
