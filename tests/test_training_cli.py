from __future__ import annotations

import base64
import gzip
import io
import json
from pathlib import Path

import pytest
import torch

from new_dt.training_cli import (
    PACKED_TOKEN_FILE,
    TOKEN_FILE,
    build_parser,
    load_prepared_corpus,
)
from new_dt.word_tokenizer import WordSpaceTokenizer


def _write_prepared_data(root: Path, *, packed: bool) -> tuple[WordSpaceTokenizer, torch.Tensor]:
    text = "alpha beta gamma delta epsilon zeta eta theta"
    tokenizer = WordSpaceTokenizer.train(text, lowercase=True, min_frequency=1)
    tokenizer.save(root / "tokenizer.json")

    ids = torch.tensor(
        tokenizer.encode_document((text + "\n") * 40),
        dtype=torch.int32,
    )
    payload = {
        "format": "new-dt-word-space-token-stream",
        "version": 1,
        "vocab_size": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_id,
        "input_ids": ids,
    }
    if packed:
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        encoded = base64.b64encode(gzip.compress(buffer.getvalue())).decode("ascii")
        (root / PACKED_TOKEN_FILE).write_text(encoded, encoding="ascii")
    else:
        torch.save(payload, root / TOKEN_FILE)

    metadata = {
        "tokenizer": {"vocab_size": tokenizer.vocab_size},
        "pretraining_corpus": {
            "encoded_tokens": ids.numel(),
            "characters": len(text) * 40,
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return tokenizer, ids.to(torch.long)


@pytest.mark.parametrize("packed", [False, True])
def test_prepared_loader_preserves_exact_token_ids(tmp_path: Path, packed: bool) -> None:
    tokenizer, original = _write_prepared_data(tmp_path, packed=packed)

    corpus = load_prepared_corpus(
        tmp_path,
        validation_fraction=0.2,
        seq_len=8,
    )

    reconstructed = torch.cat((corpus.train_tokens, corpus.validation_tokens))
    assert corpus.tokenizer.tokens == tokenizer.tokens
    assert corpus.tokenizer.lowercase is True
    assert torch.equal(reconstructed, original)
    assert corpus.train_tokens.dtype == torch.long
    assert corpus.validation_tokens.dtype == torch.long


def test_prepared_loader_rejects_tokenizer_mismatch(tmp_path: Path) -> None:
    tokenizer, _ = _write_prepared_data(tmp_path, packed=False)
    payload = torch.load(tmp_path / TOKEN_FILE, weights_only=False)
    payload["vocab_size"] = tokenizer.vocab_size + 1
    torch.save(payload, tmp_path / TOKEN_FILE)

    with pytest.raises(ValueError, match="vocabulary mismatch"):
        load_prepared_corpus(tmp_path, validation_fraction=0.2, seq_len=8)


def test_comparison_parser_accepts_prepared_data_without_raw_text() -> None:
    args = build_parser().parse_args(["--prepared-data", "data/sciq"])
    assert args.data is None
    assert args.prepared_data == Path("data/sciq")
