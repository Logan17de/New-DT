from __future__ import annotations

import json
from pathlib import Path

import torch

from new_dt import DynamicTransformer, DynamicTransformerConfig
from new_dt.comparison import main, make_batch_plan, prepare_corpus
from new_dt.small_gpt import SmallGPT
from new_dt.word_tokenizer import SPECIAL_TOKENS, WordSpaceTokenizer


def tiny_config(vocab_size: int) -> DynamicTransformerConfig:
    return DynamicTransformerConfig(
        vocab_size=vocab_size,
        d_model=8,
        n_heads=2,
        n_layers=1,
        ffn_dim=16,
        max_seq_len=8,
        dropout=0.0,
        pool_growth_factor=2.0,
        route_page_size=8,
        route_templates_per_page=2,
        route_linear_out_tile=4,
        route_lm_head_tile=8,
    )


def test_word_space_tokenizer_is_deterministic_and_persistent(tmp_path: Path) -> None:
    text = "Hello world!\nHello there"
    tokenizer = WordSpaceTokenizer.train(text, lowercase=True)

    assert tokenizer.tokens[:4] == list(SPECIAL_TOKENS)
    assert tokenizer.tokenize("Hello   world!") == ["hello", "world!"]
    assert tokenizer.encode_document(text).count(tokenizer.eos_id) == 2
    assert tokenizer.encode("missing") == [tokenizer.unk_id]

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = WordSpaceTokenizer.load(path)
    assert restored.tokens == tokenizer.tokens
    assert restored.encode_document(text) == tokenizer.encode_document(text)


def test_shared_gpt_matches_topology_and_keeps_lm_head_untied() -> None:
    config = tiny_config(vocab_size=12)
    model = SmallGPT(config)
    batch = torch.randint(0, config.vocab_size, (2, config.max_seq_len))
    output = model(batch, labels=batch)

    assert output.logits.shape == (2, config.max_seq_len, config.vocab_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert model.lm_is_untied
    assert model.embedding.weight.data_ptr() != model.lm_head.weight.data_ptr()
    assert model.layers[0].attention.q_proj.bias is None
    assert model.layers[0].ffn.up_proj.bias is None


def test_gpt_and_sdt_accept_the_exact_same_model_config() -> None:
    config = tiny_config(vocab_size=10)
    gpt = SmallGPT(config)
    sdt = DynamicTransformer(config)
    batch = torch.randint(0, config.vocab_size, (1, config.max_seq_len))

    gpt_output = gpt(batch, labels=batch)
    sdt_output = sdt(batch, labels=batch)
    assert gpt_output.logits.shape == sdt_output.logits.shape
    assert gpt_output.loss is not None and torch.isfinite(gpt_output.loss)
    assert sdt_output.loss is not None and torch.isfinite(sdt_output.loss)
    assert (
        sdt.embedding.parameters_by_token.pool.values.data_ptr()
        != sdt.lm_head.parameters_by_token.pool.values.data_ptr()
    )


def test_batch_plan_is_identical_when_seed_is_identical(tmp_path: Path) -> None:
    data = tmp_path / "data.txt"
    data.write_text("one two three four five six seven eight\n" * 20, encoding="utf-8")
    corpus = prepare_corpus(
        [data],
        lowercase=False,
        min_frequency=1,
        max_vocab=None,
        validation_fraction=0.2,
        seq_len=8,
    )
    first = make_batch_plan(
        corpus,
        steps=3,
        grad_accum=2,
        batch_size=2,
        eval_batches=2,
        seq_len=8,
        seed=9,
    )
    second = make_batch_plan(
        corpus,
        steps=3,
        grad_accum=2,
        batch_size=2,
        eval_batches=2,
        seq_len=8,
        seed=9,
    )
    assert torch.equal(first.train_starts, second.train_starts)
    assert torch.equal(first.validation_starts, second.validation_starts)


def test_comparison_cli_smoke_trains_both_models(tmp_path: Path) -> None:
    data = tmp_path / "tiny.txt"
    data.write_text(
        "alpha beta gamma delta epsilon zeta eta theta\n" * 30,
        encoding="utf-8",
    )
    output = tmp_path / "runs"
    exit_code = main(
        [
            "--data",
            str(data),
            "--model",
            "both",
            "--output-dir",
            str(output),
            "--run-name",
            "smoke",
            "--device",
            "cpu",
            "--d-model",
            "4",
            "--heads",
            "1",
            "--layers",
            "1",
            "--ffn-dim",
            "8",
            "--seq-len",
            "4",
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--eval-batches",
            "1",
            "--eval-interval",
            "1",
            "--log-interval",
            "1",
            "--warmup-steps",
            "0",
            "--structure-interval",
            "0",
            "--route-page-size",
            "4",
            "--route-templates-per-page",
            "2",
            "--route-linear-out-tile",
            "2",
            "--route-lm-head-tile",
            "4",
            "--no-save-checkpoint",
        ]
    )
    assert exit_code == 0
    comparison = json.loads((output / "smoke" / "comparison.json").read_text())
    assert [item["model"] for item in comparison] == ["gpt", "sdt"]
    assert all(item["final_validation_ppl"] > 0 for item in comparison)
    assert (output / "smoke" / "tokenizer.json").is_file()
