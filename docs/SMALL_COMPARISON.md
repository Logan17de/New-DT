# Small shared-GPT versus sDT comparison

This experiment compares two causal language models using the same tokenizer,
pre-tokenized corpus, architecture dimensions, optimizer schedule, and exact batch
start positions.

## Models

### Shared GPT

- conventional shared token embedding;
- shared Q/K/V/O matrices per layer;
- shared SwiGLU Up/Gate/Down matrices per layer;
- RMSNorm and RoPE;
- causal scaled-dot-product attention;
- untied LM head.

### sDT

The sDT model uses the same hidden size, heads, depth, FFN width, RMSNorm, RoPE,
SwiGLU, causal attention, dropout, and untied output design. Its embedding,
attention, FFN, and LM values are reconstructed from token-owned scalar routes.

```text
shared GPT → conventional shared attention/FFN matrices
sDT       → token-owned scalar-routed attention/FFN matrices
```

## Canonical SciQ input

Prepare the dataset once:

```bash
pip install -e ".[data]"
new-dt-prepare-sciq \
  --output data/sciq \
  --lowercase \
  --min-frequency 2
```

Then train from the saved artifacts:

```bash
new-dt-compare \
  --prepared-data data/sciq \
  --model both \
  --run-name sciq_static \
  --device cuda \
  --d-model 32 \
  --heads 4 \
  --layers 2 \
  --ffn-dim 128 \
  --seq-len 64 \
  --batch-size 8 \
  --steps 1000 \
  --lr 3e-4 \
  --structure-interval 0
```

`--prepared-data` loads `tokenizer.json` and `pretrain_train_tokens.pt` directly.
The command does not call `split()`, rebuild vocabulary IDs, or retokenize text.
Before training, it validates:

- prepared format/version;
- vocabulary size;
- EOS ID;
- metadata token count;
- integer dtype and ID range;
- minimum stream length for the chosen sequence length.

A repository-local `data/sciq` directory is auto-selected when neither
`--prepared-data` nor `--data` is supplied.

## Static and dynamic comparisons

Static routing isolates the scalar-routed representation:

```bash
new-dt-compare --prepared-data data/sciq --model both --structure-interval 0
```

Dynamic routing enables split and merge:

```bash
new-dt-compare \
  --prepared-data data/sciq \
  --model both \
  --structure-interval 100 \
  --min-owner-samples 4 \
  --max-splits-per-pass 4 \
  --max-merges-per-pass 4
```

The current merge index requires zero weight decay. With nonzero weight decay,
pass `--no-merge`.

## Raw custom text

Raw text remains available for unrelated experiments:

```bash
new-dt-compare --data part1.txt part2.txt --model both
```

That mode deliberately trains a new whitespace tokenizer. It is mutually exclusive
with `--prepared-data`.

## Fairness controls

For `--model both`, the runner resets the seed before each model and reuses the
same precomputed batch plan. Both models receive:

- identical tokenizer IDs;
- identical train/validation token streams;
- identical sequence length and batch order;
- identical AdamW settings and cosine schedule;
- identical gradient accumulation and clipping;
- identical RMSNorm, RoPE, SwiGLU, causal attention, and untied-output topology.

The models are not artificially parameter-count matched. Reports include normal
parameter count, sDT active scalar count, effective active parameters, logical
route references, estimated packed route storage, throughput, and perplexity.

## Output

```text
runs/small_comparison/<run>/
├── tokenizer.json
├── run_config.json
├── comparison.json
├── gpt/
│   ├── metrics.jsonl
│   ├── summary.json
│   └── checkpoint.pt
└── sdt/
    ├── metrics.jsonl
    ├── summary.json
    └── checkpoint.pt
```

`run_config.json` records whether the source was `prepared` or `raw_text` and the
exact tokenizer/token-stream paths. Use `--no-save-checkpoint` for metric-only
experiments and `--export-routing` to save final packed sDT routes.
