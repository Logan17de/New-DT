# Direct FFN MOD-4 placement benchmark

This benchmark runs three paired follow-ups to the completed
`Shared ATTN + unique FFN` SciQ model. All three use the same `d_model=32`,
three-layer, FFN-width-64, 30,000-step protocol.

## Direct MOD definition

Each token owns one four-value vector shared across every Transformer layer:

```text
m[t] in R^4
```

There is no learned projection matrix. Because the FFN width is 64, the four
values are tiled sixteen times with a fixed, parameter-free operation:

```text
[m0, m1, m2, m3]
    ->
[m0, m1, m2, m3, m0, m1, m2, m3, ...]  # width 64
```

Call the resulting direct FFN-width vector `M[t]`.

## Model 1: branch before activation

```text
up   = W_up[t,l] x
 gate = W_gate[t,l] x
act  = SiLU(gate) * (up + alpha M[t])
out  = W_down[t,l] act
```

The MOD changes the Up/content branch before SwiGLU multiplication.

## Model 2: gate before activation

```text
up   = W_up[t,l] x
gate = W_gate[t,l] x
act  = SiLU(gate + alpha M[t]) * up
out  = W_down[t,l] act
```

The MOD changes the token-specific gate before the SiLU nonlinearity.

## Model 3: after activation

```text
up   = W_up[t,l] x
gate = W_gate[t,l] x
act  = SiLU(gate) * up + alpha M[t]
out  = W_down[t,l] act
```

The MOD adds directly to the completed SwiGLU activation before the token-unique
Down projection.

## Paired initialization

Every variant first constructs the complete existing
`SharedAttentionUniqueFFN` baseline. Only afterward does it add the
zero-initialized token MOD table.

With the same seed:

- every baseline parameter is exactly identical;
- the MOD contribution is exactly zero;
- all three variants and the prior baseline have identical step-zero logits and
  loss;
- all three use the same SparseAdam/AdamW optimizer split.

The only architectural difference among the three new runs is the direct MOD
placement.

## Parameters

For vocabulary 26,466 and MOD width 4:

```text
Shared ATTN + unique FFN baseline    489,527,648
One shared direct MOD table              105,864
Learned MOD projections                         0
------------------------------------------------
Each direct-MOD model                489,633,512
```

The MOD adds 0.0216% parameters to the paired baseline.

## Locked run

```bash
new-dt-direct-ffn-mod-trio \
  --prepared-data data/sciq.zip \
  --run-name sciq_direct_mod4_placements_d32_l3_30k \
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

The command always runs all three models sequentially. It reuses the same fixed
training batch plan, validation starts, learning-rate schedule, live dashboard,
5,000-step static snapshots, sustained-overfitting detector, and final report
format as the controlled five-model benchmark.
