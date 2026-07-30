# New-DT: Scalar-Pool Dynamic Transformer

New-DT is a research implementation in which every vocabulary token owns exact
routes into independent scalar-neuron pools. A scalar may be reused by unrelated
tokens at unrelated route positions. Adam updates each unique scalar once, while
New-DT retains route-slot gradient evidence for dynamic split and merge.

## Architecture

```text
Input token IDs
    │
    ├── SPRC routes → Embedding scalar pool
    ▼
Shared residual stream
    │
    ├── Shared RMSNorm
    ├── SPRC routes → Q/K/V/O scalar pools
    ├── RoPE rotations on Q/K
    ├── shared causal mask + softmax
    └── shared residual addition
    │
    ├── Shared RMSNorm
    ├── SPRC routes → Up/Gate/Down scalar pools
    └── shared residual addition
    ▼
Shared final RMSNorm
    │
    └── SPRC routes → untied LM scalar pool
        ▼
      logits
```

Embedding, every attention projection, every FFN projection, and LM output use
separate scalar pools. Residuals, normalization, topology, masking, softmax, and
RoPE remain shared communication mechanics.

## Selective Page Reconstruction Compression

Dense `vocab_size × route_size` route tensors are replaced by SPRC:

```text
token + page
    → compact adaptive template selector
    → immutable base template
    → optional shared sparse delta
    → rare token-specific exceptions
    → exact resolved neuron IDs
```

Pages are storage and selective-decoding units only. Scalar sharing still occurs
at arbitrary individual route slots.

### Structural updates

- A split clones one scalar and writes an exact route exception.
- Reused complete patches become shared deltas.
- Large patches are promoted to immutable templates.
- Merges rewrite only exact owners and can make old templates reclaimable.
- Reverse owners are derived from template references, compact selectors, and
  sparse deviations instead of a dense second edge table.

### Optimized execution

- Adaptive per-token/page selectors
- Immutable template/delta LRU cache
- Grouped batch page reconstruction
- Tiled Q/K/V/O and FFN reconstruction
- Tiled untied LM-head reconstruction
- Exact route-slot gradient capture across tiles
- Packed arbitrary-width integer containers
- Memory-mapped selective page reads
- Cached RoPE cosine/sine tensors

See [docs/SPRC.md](docs/SPRC.md) for the complete design.

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Run the synthetic demo and routing benchmark:

```bash
python train_demo.py --steps 20 --grad-accum 2
python benchmarks/benchmark_sprc.py
```

## Prepare SciQ once

The preparation command downloads SciQ, trains the word-space tokenizer from the
training split only, and writes the exact pre-tokenized support corpus:

```bash
pip install -e ".[data]"
new-dt-prepare-sciq \
  --output data/sciq \
  --lowercase \
  --min-frequency 2
```

Generated canonical inputs include:

```text
data/sciq/
├── tokenizer.json
├── pretrain_train_tokens.pt
├── metadata.json
├── qa_train.jsonl
├── qa_validation.jsonl
└── qa_test.jsonl
```

See [docs/SCIQ_PREPARATION.md](docs/SCIQ_PREPARATION.md).

## Small GPT versus sDT comparison

Use the **saved tokenizer and saved token IDs directly**:

```bash
python compare_small.py \
  --prepared-data data/sciq \
  --model both \
  --run-name sciq_static \
  --device cuda \
  --d-model 32 \
  --heads 4 \
  --layers 2 \
  --ffn-dim 128 \
  --seq-len 64 \
  --steps 1000 \
  --batch-size 8 \
  --structure-interval 0
```

The installed command is equivalent:

```bash
new-dt-compare --prepared-data data/sciq --model both ...
```

When `data/sciq` exists, `--prepared-data` may be omitted. The runner then selects
that directory automatically. It verifies the tokenizer vocabulary size, EOS ID,
token count, integer range, metadata, and token-stream format before constructing
either model. It does **not** rebuild or retokenize the corpus.

For a dynamic sDT run:

```bash
new-dt-compare \
  --prepared-data data/sciq \
  --model both \
  --run-name sciq_dynamic \
  --d-model 32 \
  --heads 4 \
  --layers 2 \
  --ffn-dim 128 \
  --seq-len 64 \
  --steps 1000 \
  --structure-interval 100
```

Raw custom text remains supported with `--data file1.txt file2.txt`; that mode
intentionally trains a new tokenizer. `--data` and `--prepared-data` cannot be used
together.

Both comparison models use the same:

- exact tokenizer IDs and precomputed token stream;
- train/validation split and batch-start plan;
- RMSNorm, RoPE, SwiGLU, causal attention, depth, width, and dropout;
- AdamW settings, LR schedule, accumulation, and clipping;
- untied embedding and LM-head storage.

The intended difference is conventional shared Q/K/V/O and FFN matrices versus
sDT token-owned scalar-routed matrices.

See [docs/SMALL_COMPARISON.md](docs/SMALL_COMPARISON.md) for all controls and
output files.

## Minimal usage

```python
import torch
from new_dt import DynamicStructureController, DynamicTransformer, DynamicTransformerConfig

config = DynamicTransformerConfig(
    vocab_size=32,
    d_model=8,
    n_heads=2,
    n_layers=1,
    ffn_dim=16,
    route_page_size=1024,
    route_templates_per_page=16,
)
model = DynamicTransformer(config)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
controller = DynamicStructureController(structure_interval=100)

input_ids = torch.randint(0, config.vocab_size, (2, 12))
output = model(input_ids, labels=input_ids, collect_route_grads=True)
output.loss.backward()
controller.collect(model)
optimizer.step()
controller.maybe_restructure(model, optimizer, optimizer_step=1)
optimizer.zero_grad(set_to_none=True)
```

## Implementation status

The repository implements exact SPRC semantics, adaptive selectors, immutable
templates, shared deltas, exceptions, compaction, reverse queries, packed storage,
mmap selective reads, tiled execution, RoPE, split/merge integration, a matched
small-GPT baseline, SciQ preparation, and exact prepared-token training. Native
fused CUDA decoding and distributed route sharding remain optional future backends.
