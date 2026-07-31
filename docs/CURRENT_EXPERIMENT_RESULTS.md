# Current controlled SciQ experiment results

Updated: 2026-07-31

This document records the completed controlled language-model experiments discussed during development. Results use the prepared SciQ word-token stream, vocabulary size 26,466, sequence length 64, batch size 8, seed 42, cosine-style LR schedule with LR 3e-4 and 50 warmup steps, fixed validation starts, RMSNorm, RoPE, SwiGLU, causal attention, and an untied LM head.

## Main d=32, 3-layer benchmark

All models below use:

- `d_model=32`
- `n_layers=3`
- `n_heads=4`
- `ffn_dim=64`
- `steps=30,000`
- evaluation every 50 steps

| Rank | Architecture | Best validation PPL | Best step | Overfit onset | Parameters |
|---:|---|---:|---:|---:|---:|
| 1 | Shared GPT | **333.474** | 19,300 | 19,350 | 1,724,768 |
| 2 | Shared Attention + token-unique FFN | **343.010** | 25,500 | 26,950 | 489,527,648 |
| 3 | Unique Attention + shared FFN | 420.224 | 6,250 | 6,650 | 326,926,688 |
| 4 | Unique Attention + shared FFN + projected post-activation FFN MOD-4 | 420.496 | 6,700 | 8,500 | 327,033,320 |
| 5 | Fully token-unique Attention + FFN | 481.663 | 6,800 | 7,700 | 814,729,568 |

### Main finding

At one layer and width 16, token-unique Attention had performed better than token-unique FFN. At three layers and width 32, the result reversed strongly: shared Attention with token-unique FFN generalized much better. This is evidence that the value of token specialization depends on depth and scale. It is not proof that unique FFN is universally superior.

## Projected Attention MOD experiment

Architecture:

```text
Shared Attention
+ token-specific projected Attention MOD-4
→ token-unique FFN
```

| Best validation PPL | Best step | Overfit onset | Parameters |
|---:|---:|---:|---:|
| 345.222 | 21,500 | 21,750 | 489,633,896 |

This did not improve the shared-Attention + unique-FFN baseline. The static token-conditioned Attention bias likely interfered with context-dependent shared Attention features.

## Direct FFN MOD placement experiment

All three variants use one sparse token MOD table shared across all layers, no learned projection, and a parameter-free repeat/tile from MOD width to FFN width.

For MOD-4 and FFN width 64:

```text
[m0, m1, m2, m3] repeated 16 times → width 64
```

### Equations

Branch/Up before multiplication:

```text
a = SiLU(gate) * (up + MOD)
```

Gate before SiLU:

```text
a = SiLU(gate + MOD) * up
```

After SwiGLU activation:

```text
a = SiLU(gate) * up + MOD
```

### MOD-4 results

| Rank | Placement | Best validation PPL | Best step | Overfit onset | Final PPL | Parameters |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **Gate before SiLU** | **342.835** | 21,550 | 21,800 | 348.780 | 489,633,512 |
| 2 | Branch/Up before multiplication | 343.147 | 21,500 | 21,800 | — | 489,633,512 |
| 3 | After activation | 343.888 | 23,150 | 23,650 | 348.630 | 489,633,512 |

The Gate placement was the only MOD-4 variant that improved on the no-MOD baseline (`343.010`), but the gain was only `0.175` PPL, approximately `0.051%`. This must be repeated across seeds before treating the gain as robust.

## Gate MOD dimension scaling

The winning Gate placement was repeated with MOD dimension 8 (`4 × 2`). There is still no learned projection. The eight values repeat eight times to fill FFN width 64.

| MOD width | Best validation PPL | Best step | Overfit onset | Final PPL | Final regression | Parameters |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 342.835 | 21,550 | 21,800 | 348.780 | ~1.73% | 489,633,512 |
| **8** | **342.6633** | **21,550** | **21,800** | **348.0033** | **1.56%** | **489,739,376** |

MOD-8 improved over MOD-4 by `0.1717` PPL, approximately `0.050%`. The identical best step and overfit onset indicate that doubling MOD capacity did not materially alter optimization dynamics. The result suggests a small capacity benefit, with likely diminishing returns.

## Current ranking by best PPL

| Rank | Model | Best validation PPL |
|---:|---|---:|
| 1 | Shared GPT | **333.474** |
| 2 | Shared Attention + unique FFN + direct Gate MOD-8 | **342.6633** |
| 3 | Shared Attention + unique FFN + direct Gate MOD-4 | 342.835 |
| 4 | Shared Attention + unique FFN | 343.010 |
| 5 | Shared Attention + unique FFN + direct Branch MOD-4 | 343.147 |
| 6 | Shared Attention + unique FFN + direct post-activation MOD-4 | 343.888 |
| 7 | Shared Attention + projected Attention MOD-4 + unique FFN | 345.222 |
| 8 | Unique Attention + shared FFN | 420.224 |
| 9 | Unique Attention + shared FFN + projected FFN MOD-4 | 420.496 |
| 10 | Fully unique Attention + FFN | 481.663 |

## Overfitting and dashboard behavior

The benchmark dashboard reports:

- current and best validation PPL;
- best step;
- sustained overfit onset;
- train PPL;
- throughput;
- parameter count;
- progress and ETA;
- CUDA allocation/reservation.

Sustained overfitting is defined as the first evaluation in a three-evaluation run at least 0.5% above the best PPL while training loss does not worsen. Direct-MOD runs additionally stop if validation PPL exceeds the best PPL by more than 5%. Static snapshots are emitted every 5,000 steps. In Colab, the notebook-native runner redraws the live dashboard with `clear_output(wait=True)` so intermediate live frames do not accumulate.

## Scientific cautions

- Most comparisons currently use one seed.
- The large lookup models use SparseAdam for token-owned rows and AdamW for shared dense parameters, while GPT uses dense AdamW.
- Parameter counts differ dramatically.
- Best-checkpoint quality and final-step stability are separate measurements.
- Small PPL differences such as MOD-4 versus MOD-8 need multi-seed confirmation.

## Recommended next checks

1. Repeat the no-MOD baseline, Gate MOD-4, and Gate MOD-8 with at least three seeds.
2. Test Gate MOD-16 only after multi-seed confirmation of the 4→8 improvement.
3. Preserve shared Attention and investigate more parameter-efficient alternatives to full token-unique FFN.
