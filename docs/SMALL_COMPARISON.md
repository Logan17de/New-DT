# Small shared-GPT versus sDT comparison

This experiment compares two causal language models using one dataset, one
word-space tokenizer, one architecture configuration, one optimizer schedule, and
exactly the same precomputed train/evaluation batches.

## Models

### `gpt`

A conventional small GPT-style model with shared matrices:

- one shared token embedding table;
- shared Q/K/V/O matrices per layer;
- shared SwiGLU Up/Gate/Down matrices per layer;
- RMSNorm;
- RoPE;
- causal scaled-dot-product attention;
- an **untied** LM head.

### `sdt`

The New-DT model with the same hidden size, heads, layers, FFN width, RMSNorm,
RoPE, SwiGLU, causal attention, dropout, and untied LM head. Its embedding,
attention, FFN, and output values are reconstructed from scalar-neuron routes.

The intended model difference is therefore:

```text
shared GPT: one conventional attention/FFN matrix per layer
sDT:       token-owned scalar-routed attention/FFN matrices
```

## Word-space tokenizer

The tokenizer is trained directly from the supplied UTF-8 text files.

```text
text.split()
```

It splits only on whitespace. Punctuation remains attached to words, so `model.`
and `model` are different tokens. The vocabulary is ordered deterministically by
frequency and then alphabetically. The first IDs are always:

```text
<pad> <unk> <bos> <eos>
```

Every non-empty dataset line receives one `<eos>` token.

## Basic run

```bash
python compare_small.py \
  --data dataset.txt \
  --model both \
  --d-model 32 \
  --heads 4 \
  --layers 2 \
  --ffn-dim 128 \
  --seq-len 64 \
  --batch-size 8 \
  --steps 500 \
  --lr 3e-4 \
  --structure-interval 100
```

The installed CLI is equivalent:

```bash
new-dt-compare --data dataset.txt --model both ...
```

Multiple files may be supplied:

```bash
python compare_small.py --data part1.txt part2.txt part3.txt --model both
```

## Strict static-routing comparison

Disable sDT split/merge to measure only the scalar-routed representation:

```bash
python compare_small.py \
  --data dataset.txt \
  --model both \
  --structure-interval 0
```

## Dynamic sDT comparison

Enable structural changes and tune them from the CLI:

```bash
python compare_small.py \
  --data dataset.txt \
  --model both \
  --structure-interval 50 \
  --min-owner-samples 4 \
  --min-conflict-score 0.5 \
  --max-splits-per-pass 4 \
  --max-merges-per-pass 4
```

The current merge index requires zero weight decay. With nonzero weight decay,
pass `--no-merge`.

## Recommended first configuration

```bash
python compare_small.py \
  --data dataset.txt \
  --model both \
  --run-name first_test \
  --device cuda \
  --d-model 32 \
  --heads 4 \
  --layers 2 \
  --ffn-dim 128 \
  --seq-len 64 \
  --dropout 0 \
  --steps 1000 \
  --batch-size 8 \
  --grad-accum 1 \
  --lr 3e-4 \
  --warmup-steps 50 \
  --eval-interval 50 \
  --eval-batches 20 \
  --initial-shared-fraction 0.5 \
  --route-page-size 256 \
  --route-templates-per-page 4 \
  --structure-interval 100
```

Keeping dropout at zero makes the comparison easier to reproduce because the two
architectures consume random numbers differently internally.

## Fairness controls

For `--model both`, the runner resets the seed before constructing each model and
reuses the same tensors containing every train and validation start position.
Both models therefore receive:

- the same tokenizer and vocabulary;
- the same train/validation token split;
- the same sequence length and batches in the same order;
- the same AdamW hyperparameters and cosine schedule;
- the same gradient accumulation and clipping;
- the same RMSNorm, RoPE, SwiGLU, causal attention, and untied output design.

The models are intentionally **not parameter-count matched**. The report includes
normal parameter count, sDT active scalar count, effective active parameters,
logical route references, and estimated packed route storage so quality can be
interpreted together with capacity and cost.

## Output

Each run creates:

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

Use `--no-save-checkpoint` for metric-only experiments. Use `--export-routing` to
write packed `.sprc` route containers for the final sDT model.

The terminal comparison reports:

- final and best validation perplexity;
- training tokens per second;
- trainable parameters;
- sDT effective active parameters;
- estimated packed route storage;
- split and merge counts in the JSON summary.
