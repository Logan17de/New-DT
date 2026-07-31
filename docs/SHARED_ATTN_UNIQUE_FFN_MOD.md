# Shared Attention + token-unique FFN + projected MOD

This is a single-model follow-up to the controlled five-model SciQ benchmark.
It is intended to be compared directly with the completed
`shared_attn_unique_ffn` result from the same `d_model=32`, three-layer,
30,000-step protocol.

## Architecture

For token `t` in layer `l`:

```text
shared Q/K/V/O attention
        ↓
token-unique Up and Gate projections
        ↓
SwiGLU activation
        ↓
+ layer_projection(shared_token_mod[t])
        ↓
token-unique Down projection
```

Mathematically:

```text
a[t,l]  = SiLU(W_gate[t,l] x[t,l]) * W_up[t,l] x[t,l]
Δ[t,l]  = m[t] P[l]
a'[t,l] = a[t,l] + α Δ[t,l]
y[t,l]  = W_down[t,l] a'[t,l]
```

There is one small token MOD table `m[t]` shared across all Transformer layers.
Every layer owns a separate projection `P[l]` from MOD width to FFN width.

## Strong paired initialization

The implementation first constructs the complete existing
`SharedAttentionUniqueFFN` baseline. Only after every baseline parameter and the
LM head have been initialized does it add the zero-initialized token MOD table and
layer projections.

With the same seed:

- all baseline weights are exactly identical;
- the MOD contribution is exactly zero;
- step-zero logits and loss are exactly identical.

This makes the follow-up a cleaner test of the projected MOD than comparing two
independently initialized architectures.

## Locked run

```bash
new-dt-shared-attn-unique-ffn-mod \
  --prepared-data data/sciq.zip \
  --run-name sciq_shared_attn_unique_ffn_mod4_d32_l3_30k \
  --device cuda \
  --seed 42 \
  --d-model 32 \
  --heads 4 \
  --layers 3 \
  --ffn-dim 64 \
  --ffn-mod-dim 4 \
  --ffn-mod-scale 1.0 \
  --seq-len 64 \
  --dropout 0 \
  --steps 30000 \
  --batch-size 8 \
  --grad-accum 1 \
  --lr 3e-4 \
  --warmup-steps 50 \
  --min-lr-ratio 0.1 \
  --eval-interval 50 \
  --eval-batches 20 \
  --dashboard-interval 10 \
  --static-log-interval 5000 \
  --overfit-threshold 0.005 \
  --overfit-patience 3 \
  --grad-clip 1.0 \
  --weight-decay 0 \
  --structure-interval 0 \
  --no-save-checkpoint
```

The command reuses the same dashboard, fixed batch-plan generation, validation
starts, learning-rate schedule, overfitting detector, metrics files, and report
format as the five-model benchmark.

## Expected SciQ parameter count

For vocabulary 26,466, `d_model=32`, three layers, FFN width 64, and MOD width 4:

```text
Shared ATTN + unique FFN baseline    489,527,648
One shared token MOD table               105,864
Three layer projections                       768
------------------------------------------------
Total                                  489,634,280
```

## Interpretation boundary

The five-model benchmark is a controlled end-to-end comparison of each
architecture and its feasible optimizer stack. GPT uses dense AdamW. Token-owned
lookup rows use SparseAdam while shared parameters use AdamW. Therefore the sweep
is not a pure optimizer-independent architecture proof.

This new run is particularly strong against `Shared ATTN + unique FFN`, because
both models use the same sparse/dense optimizer split and the new model preserves
the exact baseline initialization at step zero.
