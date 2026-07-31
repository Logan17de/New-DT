# Shared Attention + token Attention MOD + token-unique FFN

This is the corrected single-model follow-up to the controlled five-model SciQ
benchmark. It compares directly with the completed `shared_attn_unique_ffn` run
using the same `d_model=32`, three-layer, 30,000-step protocol.

## Architecture

For token `t` in layer `l`:

```text
shared Q/K/V/O attention output
        +
layer_projection(shared_token_mod[t])
        ↓
attention residual
        ↓
token-unique SwiGLU Up/Gate/Down FFN
```

Mathematically:

```text
A[t,l]  = SharedAttention_l(Norm(h[t,l]))
D[t,l]  = alpha * m[t] P_attn[l]
h'[t,l] = h[t,l] + Dropout(A[t,l] + D[t,l])
y[t,l]  = h'[t,l] + Dropout(UniqueFFN_l(Norm(h'[t,l]), token=t))
```

The token MOD table is shared across all Transformer layers:

```text
m[t] in R^d_mod
```

Every layer owns a separate projection into model width:

```text
P_attn[l] in R^(d_mod x d_model)
```

The MOD is not placed inside the FFN and does not project to FFN width.

## Strong paired initialization

The implementation first constructs the complete existing
`SharedAttentionUniqueFFN` baseline. Only after all baseline parameters and the LM
head are initialized does it add the zero-initialized token MOD table and the
layer-specific Attention-MOD projections.

With the same seed:

- every baseline parameter is exactly identical;
- the Attention MOD contribution is exactly zero;
- step-zero logits and loss are exactly identical.

This isolates the effect of the new Attention MOD path much more cleanly than an
independently initialized comparison.

## Locked run

```bash
new-dt-shared-attn-mod-unique-ffn \
  --prepared-data data/sciq.zip \
  --run-name sciq_shared_attn_mod4_unique_ffn_d32_l3_30k \
  --device cuda \
  --seed 42 \
  --d-model 32 \
  --heads 4 \
  --layers 3 \
  --ffn-dim 64 \
  --attn-mod-dim 4 \
  --attn-mod-scale 1.0 \
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

The command reuses the same fixed batch-plan generation, validation starts,
learning-rate schedule, sparse/dense optimizer split, dashboard, static snapshots,
overfitting detector, and final report format as the five-model benchmark.

## Expected SciQ parameter count

For vocabulary 26,466, `d_model=32`, three layers, FFN width 64, and MOD width 4:

```text
Shared ATTN + unique FFN baseline    489,527,648
One shared token MOD table               105,864
Three d_mod -> d_model projections            384
------------------------------------------------
Total                                  489,633,896
```

The Attention MOD adds 106,248 parameters, approximately 0.0217% over the paired
baseline.

## Comparison boundary

The original five-model sweep is a controlled end-to-end comparison of complete
architectures and their feasible optimizer stacks. GPT uses dense AdamW, while
token-owned lookup rows use SparseAdam and shared parameters use AdamW.

This new run is a particularly strong paired comparison against
`Shared ATTN + unique FFN` because both use the same optimizer split, same batch
plan, and exactly identical baseline initialization at step zero.
