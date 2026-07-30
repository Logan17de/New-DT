# Direct lookup GPT versus DT comparison

This small experiment contains exactly two models.

| Model | Embedding | Attention | FFN | LM head |
| --- | --- | --- | --- | --- |
| GPT | token rows | one shared Q/K/V/O set per layer | one shared SwiGLU set per layer | untied output-token rows |
| DT | independent token rows | independent Q/K/V/O matrices for every token | independent Up/Gate/Down matrices for every token | untied output-token rows |

DT uses no scalar pool, route sharing, SPRC, split, merge, compaction, or structural controller. The original full DynamicTransformer remains available elsewhere in the repository; it is not used by `new-dt-compare`.

## Why sparse lookup gradients are used

The parameter tables are dense persistent lookup tables, but a batch references only the rows belonging to tokens present in that batch. `nn.Embedding(..., sparse=True)` preserves that access pattern for embedding, attention, and FFN tables. The trainer uses SparseAdam for these active rows and AdamW for the dense RMSNorm and untied output-table parameters.

The LM output table is necessarily evaluated against every candidate token to produce full-vocabulary logits, so it remains dense for both models.

## Parameter scaling

For vocabulary `V`, width `D`, FFN width `F`, and depth `L`:

```text
GPT = 2VD + L(4D² + 3DF + 2D) + D
DT  = 2VD + L[V(4D² + 3DF) + 2D] + D
```

With the SciQ tokenizer (`V = 26,466`) and the safe starting configuration:

```text
D = 16
F = 32
L = 1
heads = 4
```

GPT has about 0.85 million parameters and DT has about 68.6 million parameters. The CLI prints exact counts and an approximate FP32 parameter-plus-optimizer-state footprint before constructing either model.

## Colab run

```bash
!git -C /content/New-DT pull origin main
%cd /content/New-DT
!pip install -q -e .

!new-dt-compare \
  --prepared-data data/sciq.zip \
  --model both \
  --run-name sciq_gpt_vs_lookup_dt \
  --device cuda \
  --d-model 16 \
  --heads 4 \
  --layers 1 \
  --ffn-dim 32 \
  --seq-len 64 \
  --dropout 0 \
  --steps 1000 \
  --batch-size 8 \
  --grad-accum 1 \
  --lr 3e-4 \
  --warmup-steps 50 \
  --eval-interval 50 \
  --eval-batches 20 \
  --weight-decay 0
```

There is deliberately no `--structure-interval` in this run. A nonzero value is rejected because direct lookup DT has no split/merge mode.

## Larger configurations

The default safety ceiling is 300 million DT parameters. The CLI asks you to reduce width, FFN width, or depth before an accidental oversized allocation. `--allow-large-dt` bypasses the guard when the hardware budget has been checked manually.
