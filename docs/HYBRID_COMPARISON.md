# Hybrid attention/FFN comparison

This experiment contains exactly two models:

| Model | Attention | FFN | Embedding | LM head |
| --- | --- | --- | --- | --- |
| `shared_attn_unique_ffn` | one shared Q/K/V/O set per layer | independent token-owned SwiGLU Up/Gate/Down matrices | sparse token table | dense untied output-token table |
| `unique_attn_shared_ffn` | independent token-owned Q/K/V/O matrices | one shared SwiGLU Up/Gate/Down set per layer | sparse token table | dense untied output-token table |

Both models use the same RMSNorm, RoPE, residual topology, initialization, tokenizer, token stream, batch starts, optimizer hyperparameters, schedule, validation batches, and dimensions. There is no SPRC, route sharing, split, merge, compaction, or structural controller.

## Why this test matters

The prior independent lookup DT made both attention and FFN token-private. This ablation isolates which component benefits from specialization:

- Shared attention + unique FFN tests whether FFN capacity can specialize while attention preserves transferable routing/context behavior.
- Unique attention + shared FFN tests whether token-specific routing/context behavior helps while FFN knowledge remains transferable.

## Optimizers and clipping

The token embedding and token-owned component use sparse gradients and SparseAdam. The shared component, RMSNorm, and dense untied LM head use AdamW. Weight decay must remain zero.

Sparse and dense parameter groups are gradient-clipped separately using the same `--grad-clip` value. This prevents the dense full-vocabulary output head from scaling down the token-owned component's update.

## SciQ Colab run

```bash
%cd /content/New-DT
!git checkout main
!git pull origin main
!pip install -q -e .

!new-dt-hybrid-compare \
  --prepared-data data/sciq.zip \
  --model both \
  --run-name sciq_hybrid_attn_ffn \
  --device cuda \
  --d-model 16 \
  --heads 4 \
  --layers 1 \
  --ffn-dim 32 \
  --seq-len 64 \
  --dropout 0 \
  --steps 30000 \
  --batch-size 8 \
  --grad-accum 1 \
  --lr 3e-4 \
  --warmup-steps 50 \
  --eval-interval 50 \
  --eval-batches 20 \
  --log-interval 10 \
  --grad-clip 1.0 \
  --weight-decay 0
```

For the SciQ vocabulary of 26,466 tokens with `D=16`, `F=32`, and one layer, the expected parameter counts are approximately:

```text
shared attention + unique FFN : 41.50 million
unique attention + shared FFN : 27.95 million
```

The output directory contains independent metrics, summaries, and checkpoints for both hybrids plus one shared tokenizer and run configuration.
