from __future__ import annotations

import json
from pathlib import Path

import torch

from new_dt.prepare_sciq import (
    build_pretraining_text,
    build_tokenizer_training_text,
    clean_text,
    prepare_sciq_dataset,
)
from new_dt.word_tokenizer import WordSpaceTokenizer


def _row(
    support: str,
    question: str,
    answer: str,
    distractors: tuple[str, str, str] = ("wrong one", "wrong two", "wrong three"),
) -> dict[str, str]:
    return {
        "support": support,
        "question": question,
        "correct_answer": answer,
        "distractor1": distractors[0],
        "distractor2": distractors[1],
        "distractor3": distractors[2],
    }


def test_text_builders_clean_and_deduplicate_support() -> None:
    rows = [
        _row("  clean   science text ", "what is science?", "knowledge"),
        _row("clean science text", "another question?", "another answer"),
    ]
    assert clean_text(" a\n b\t c ") == "a b c"
    assert build_pretraining_text(rows) == "clean science text"
    tokenizer_text = build_tokenizer_training_text(rows)
    assert "what is science?" in tokenizer_text
    assert "another answer" in tokenizer_text


def test_prepare_sciq_writes_reusable_tokenizer_and_stream(tmp_path: Path) -> None:
    dataset = {
        "train": [
            _row("Water freezes at zero degrees.", "When does water freeze?", "zero"),
            _row("Water freezes at zero degrees.", "What freezes?", "water"),
            _row("Earth orbits the Sun.", "What does Earth orbit?", "the Sun"),
        ],
        "validation": [
            _row("Validation-only fact.", "What is held out?", "fact"),
        ],
        "test": [
            _row("Test-only fact.", "What is tested?", "fact"),
        ],
    }

    metadata = prepare_sciq_dataset(
        dataset,
        tmp_path,
        lowercase=True,
        min_frequency=1,
    )

    assert metadata["splits"] == {"train": 3, "validation": 1, "test": 1}
    assert metadata["pretraining_corpus"]["paragraphs"] == 2
    assert (tmp_path / "pretrain_train.txt").read_text(encoding="utf-8").count(
        "water freezes"
    ) == 1

    tokenizer = WordSpaceTokenizer.load(tmp_path / "tokenizer.json")
    assert tokenizer.lowercase
    assert tokenizer.encode("validation-only")[0] == tokenizer.unk_id

    payload = torch.load(tmp_path / "pretrain_train_tokens.pt", weights_only=False)
    assert payload["vocab_size"] == tokenizer.vocab_size
    assert payload["input_ids"].dtype == torch.int32
    assert payload["input_ids"].numel() > 0

    qa = json.loads((tmp_path / "qa_validation.jsonl").read_text(encoding="utf-8"))
    assert qa["question"] == "What is held out?"
    assert qa["correct_option_index"] == 0


def test_existing_output_requires_force(tmp_path: Path) -> None:
    (tmp_path / "existing.txt").write_text("keep", encoding="utf-8")
    dataset = {
        "train": [_row("A train fact.", "Question?", "Answer")],
        "validation": [_row("A validation fact.", "Question?", "Answer")],
        "test": [_row("A test fact.", "Question?", "Answer")],
    }

    try:
        prepare_sciq_dataset(dataset, tmp_path, min_frequency=1)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected FileExistsError without force=True")
