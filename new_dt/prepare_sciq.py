from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .word_tokenizer import SPECIAL_TOKENS, WordSpaceTokenizer

DATASET_NAME = "allenai/sciq"
TEXT_FIELDS = (
    "support",
    "question",
    "correct_answer",
    "distractor1",
    "distractor2",
    "distractor3",
)
REQUIRED_SPLITS = ("train", "validation", "test")


def clean_text(value: Any) -> str:
    """Collapse Unicode whitespace without changing punctuation or wording."""

    return " ".join(str(value).split()) if value is not None else ""


def unique_nonempty(values: Iterable[Any]) -> list[str]:
    """Deduplicate cleaned strings while preserving first-seen order."""

    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = clean_text(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            output.append(cleaned)
    return output


def build_tokenizer_training_text(rows: Iterable[Mapping[str, Any]]) -> str:
    """Use every text field from the training split for vocabulary coverage."""

    lines: list[str] = []
    for row in rows:
        for field in TEXT_FIELDS:
            value = clean_text(row.get(field, ""))
            if value:
                lines.append(value)
    return "\n".join(lines)


def build_pretraining_text(rows: Iterable[Mapping[str, Any]]) -> str:
    """Create a leakage-resistant corpus from unique train support passages only."""

    return "\n".join(unique_nonempty(row.get("support", "") for row in rows))


def qa_record(row: Mapping[str, Any], row_id: int) -> dict[str, Any]:
    correct = clean_text(row.get("correct_answer", ""))
    return {
        "id": row_id,
        "support": clean_text(row.get("support", "")),
        "question": clean_text(row.get("question", "")),
        "options": [
            correct,
            clean_text(row.get("distractor1", "")),
            clean_text(row.get("distractor2", "")),
            clean_text(row.get("distractor3", "")),
        ],
        "correct_answer": correct,
        "correct_option_index": 0,
    }


def write_qa_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row_id, row in enumerate(rows):
            handle.write(json.dumps(qa_record(row, row_id), ensure_ascii=False) + "\n")
            count += 1
    return count


def oov_statistics(
    tokenizer: WordSpaceTokenizer,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, int | float]:
    total = 0
    unknown = 0
    for row in rows:
        for field in TEXT_FIELDS:
            text = clean_text(row.get(field, ""))
            ids = tokenizer.encode(text)
            total += len(ids)
            unknown += sum(token_id == tokenizer.unk_id for token_id in ids)
    fraction = unknown / total if total else 0.0
    return {
        "tokens": total,
        "unknown_tokens": unknown,
        "unknown_fraction": fraction,
        "unknown_percent": fraction * 100.0,
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _materialize_rows(split: Any) -> list[dict[str, Any]]:
    """Detach preparation logic from the optional datasets package."""

    return [dict(row) for row in split]


def prepare_sciq_dataset(
    dataset: Mapping[str, Any],
    output: Path,
    *,
    lowercase: bool = True,
    min_frequency: int = 2,
    max_vocab: int | None = None,
    force: bool = False,
    dataset_revision: str = "main",
) -> dict[str, Any]:
    """Prepare an already-loaded SciQ DatasetDict-like object."""

    if min_frequency <= 0:
        raise ValueError("min_frequency must be positive")
    if max_vocab is not None and max_vocab < len(SPECIAL_TOKENS):
        raise ValueError(f"max_vocab must be at least {len(SPECIAL_TOKENS)}")

    missing = [name for name in REQUIRED_SPLITS if name not in dataset]
    if missing:
        raise ValueError(f"SciQ dataset is missing splits: {missing}")

    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(
            f"{output} already contains files; pass --force to replace generated files"
        )
    output.mkdir(parents=True, exist_ok=True)

    split_rows = {name: _materialize_rows(dataset[name]) for name in REQUIRED_SPLITS}
    train_rows = split_rows["train"]

    tokenizer_training_text = build_tokenizer_training_text(train_rows)
    pretraining_text = build_pretraining_text(train_rows)
    if not tokenizer_training_text.strip():
        raise RuntimeError("SciQ tokenizer training text is empty")
    if not pretraining_text.strip():
        raise RuntimeError("SciQ support-only pretraining text is empty")

    tokenizer = WordSpaceTokenizer.train(
        tokenizer_training_text,
        lowercase=lowercase,
        min_frequency=min_frequency,
        max_vocab=max_vocab,
    )
    token_ids = tokenizer.encode_document(pretraining_text)

    tokenizer_path = output / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    (output / "vocab.txt").write_text(
        "\n".join(f"{index}\t{token}" for index, token in enumerate(tokenizer.tokens))
        + "\n",
        encoding="utf-8",
    )
    (output / "tokenizer_training.txt").write_text(
        tokenizer_training_text + "\n", encoding="utf-8"
    )
    (output / "pretrain_train.txt").write_text(
        pretraining_text + "\n", encoding="utf-8"
    )

    torch.save(
        {
            "format": "new-dt-word-space-token-stream",
            "version": 1,
            "dataset": DATASET_NAME,
            "dataset_revision": dataset_revision,
            "split": "train",
            "source": "unique non-empty support passages",
            "tokenizer_file": tokenizer_path.name,
            "vocab_size": tokenizer.vocab_size,
            "eos_id": tokenizer.eos_id,
            "input_ids": torch.tensor(token_ids, dtype=torch.int32),
        },
        output / "pretrain_train_tokens.pt",
    )

    split_sizes: dict[str, int] = {}
    oov: dict[str, dict[str, int | float]] = {}
    for split_name, rows in split_rows.items():
        split_sizes[split_name] = write_qa_jsonl(rows, output / f"qa_{split_name}.jsonl")
        oov[split_name] = oov_statistics(tokenizer, rows)

    metadata: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "dataset_revision": dataset_revision,
        "splits": split_sizes,
        "tokenizer": {
            "type": "word_space",
            "lowercase": lowercase,
            "min_frequency": min_frequency,
            "max_vocab": max_vocab,
            "vocab_size": tokenizer.vocab_size,
            "special_tokens": {
                "pad_id": tokenizer.pad_id,
                "unk_id": tokenizer.unk_id,
                "bos_id": tokenizer.bos_id,
                "eos_id": tokenizer.eos_id,
            },
        },
        "pretraining_corpus": {
            "source": "unique train support passages only",
            "paragraphs": len(pretraining_text.splitlines()),
            "characters": len(pretraining_text),
            "utf8_bytes": len(pretraining_text.encode("utf-8")),
            "encoded_tokens": len(token_ids),
            "sha256": sha256_text(pretraining_text),
        },
        "tokenizer_training_corpus": {
            "source": "all text fields from the train split only",
            "characters": len(tokenizer_training_text),
            "utf8_bytes": len(tokenizer_training_text.encode("utf-8")),
            "sha256": sha256_text(tokenizer_training_text),
        },
        "oov_statistics": oov,
        "files": {
            "tokenizer": "tokenizer.json",
            "vocabulary": "vocab.txt",
            "tokenizer_training_text": "tokenizer_training.txt",
            "pretraining_text": "pretrain_train.txt",
            "pretraining_tokens": "pretrain_train_tokens.pt",
            **{name: f"qa_{name}.jsonl" for name in REQUIRED_SPLITS},
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def download_and_prepare_sciq(
    output: Path,
    *,
    lowercase: bool = True,
    min_frequency: int = 2,
    max_vocab: int | None = None,
    force: bool = False,
    dataset_revision: str = "main",
    cache_dir: str | None = None,
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            'The optional data dependency is missing. Install with: pip install -e ".[data]"'
        ) from error

    dataset = load_dataset(
        DATASET_NAME,
        revision=dataset_revision,
        cache_dir=cache_dir,
    )
    return prepare_sciq_dataset(
        dataset,
        output,
        lowercase=lowercase,
        min_frequency=min_frequency,
        max_vocab=max_vocab,
        force=force,
        dataset_revision=dataset_revision,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download SciQ and build New-DT word-space tokenizer files."
    )
    parser.add_argument("--output", type=Path, default=Path("data/sciq"))
    parser.add_argument(
        "--lowercase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lowercase before tokenization (default: true).",
    )
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-vocab", type=int, default=None)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def _print_summary(metadata: Mapping[str, Any], output: Path) -> None:
    tokenizer = metadata["tokenizer"]
    corpus = metadata["pretraining_corpus"]
    print("SciQ preparation complete")
    print(f"output            : {output.resolve()}")
    print(f"vocabulary        : {int(tokenizer['vocab_size']):,}")
    print(f"support passages  : {int(corpus['paragraphs']):,}")
    print(f"pretraining tokens: {int(corpus['encoded_tokens']):,}")
    print(f"pretraining bytes : {int(corpus['utf8_bytes']):,}")
    for split_name in REQUIRED_SPLITS:
        stats = metadata["oov_statistics"][split_name]
        print(
            f"{split_name:10s} OOV     : "
            f"{int(stats['unknown_tokens']):,}/{int(stats['tokens']):,} "
            f"({float(stats['unknown_percent']):.3f}%)"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = download_and_prepare_sciq(
        args.output,
        lowercase=args.lowercase,
        min_frequency=args.min_frequency,
        max_vocab=args.max_vocab,
        force=args.force,
        dataset_revision=args.dataset_revision,
        cache_dir=args.cache_dir,
    )
    _print_summary(metadata, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
