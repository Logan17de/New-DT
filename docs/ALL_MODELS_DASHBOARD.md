# Controlled five-model dashboard benchmark

This runner trains the five maintained small-model architectures sequentially on the exact same prepared token stream and batch plan:

1. Shared GPT
2. Fully token-unique Attention + FFN
3. Shared Attention + token-unique FFN
4. Token-unique Attention + shared FFN
5. Token-unique Attention + shared FFN + projected post-activation MOD

The MOD model uses one small token table shared across all Transformer layers. Every layer retains its own projection from MOD width to FFN activation width.

## Locked 30K SciQ command

```bash
new-dt-all-models \
  --prepared-data data/sciq.zip \
  --model all \
  --run-name sciq_all_d32_l3_30k \
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
  --structure-interval 0
```

Add `--no-save-checkpoint` for a metrics-only sweep. This avoids writing several gigabytes of final checkpoints. The dashboard and every JSON/CSV/Markdown report are still preserved.

## Console behavior

The Rich dashboard redraws in place and shows current step, progress, training PPL, latest validation PPL, best PPL and step, detected overfitting onset, throughput, parameter count, ETA, and CUDA memory.

No permanent line is printed for every training step. A complete static dashboard snapshot is printed at steps 5,000, 10,000, 15,000, 20,000, 25,000, and 30,000 for each model.

## Overfitting definition

The runner reports a retrospective onset rather than labeling a single noisy validation point as overfitting. By default, onset is the first evaluation after the best checkpoint that begins three consecutive evaluations at least 0.5% above the best PPL while training loss does not worsen.

Both controls are configurable:

```text
--overfit-threshold 0.005
--overfit-patience 3
```

## Outputs

```text
runs/all_models_dashboard/<run>/
├── tokenizer.json
├── run_config.json
├── dashboard_state.json
├── comparison.json
├── comparison.csv
├── report.md
├── gpt/
├── direct_dt/
├── shared_attn_unique_ffn/
├── unique_attn_shared_ffn/
└── unique_attn_shared_ffn_mod/
```

Each model directory contains `metrics.jsonl`, `summary.json`, and—unless disabled—the final-step `checkpoint.pt`. The summary records best step and overfitting onset separately because the final checkpoint may be worse than the best validation checkpoint.

## Expected parameter scale for the current SciQ vocabulary

With vocabulary 26,466, `d_model=32`, three layers, FFN width 64, and MOD width 4, the analytical counts are approximately:

| Model | Parameters |
|---|---:|
| Shared GPT | 1,724,768 |
| Fully unique Attention + FFN | 814,729,568 |
| Shared Attention + unique FFN | 489,527,648 |
| Unique Attention + shared FFN | 326,926,688 |
| Unique Attention + shared FFN + MOD | 327,033,320 |

The models run sequentially, so peak GPU memory is determined by the largest individual model rather than the sum of all five. The fully unique model has an estimated FP32 parameter-and-optimizer footprint near 10 GB before activations and CUDA workspace.
