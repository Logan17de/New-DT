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
    └── SPRC routes → LM scalar pool
        ▼
      logits
```

Embedding, every attention projection, every FFN projection, and LM output use
separate scalar pools. Residuals, normalization, topology, masking, softmax, and
RoPE remain shared communication mechanics.

## Selective Page Reconstruction Compression

Dense `vocab_size × route_size` route tensors have been replaced by SPRC:

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

### Split

A split clones the source scalar and its Adam moments, then writes one exception:

```text
Template 7: offset 18 → neuron 75
Token 42 exception: offset 18 → neuron 901
```

The template never changes.

### Delta and template promotion

- Small unique changes remain exceptions.
- The same full patch reused by multiple token pages becomes one shared delta.
- A large patch is resolved once and absorbed into a new immutable template.
- Merges remove or rewrite patches and may make old templates unused.

### Exact reverse ownership

SPRC avoids a dense second copy of every relation edge. Exact owners are derived
from scalar-to-template references, compact selectors, and sparse deviations.
Template-user lists are built lazily and cached only when split/merge needs them.

### Optimized execution

- Per-token selector defaults replace the dense vocabulary-by-page selector tensor.
- Sparse page selector divergence promotes to a dense page only when worthwhile.
- Immutable template+delta pages use an LRU cache.
- Batch tokens are grouped by page recipe and decoded once per group.
- Q/K/V/O and FFN matrices are reconstructed in output-row tiles.
- The LM head is reconstructed in vocabulary tiles.
- Route gradients retain their exact original slots across every tile.
- RoPE cosine/sine tensors are cached by sequence length, device, and dtype.

### Packed files and selective reads

```python
routed.export_packed("embedding.sprc")

from new_dt import PackedSPRCReader
with PackedSPRCReader("embedding.sprc") as reader:
    page = reader.resolve_page(token_id=732, page_id=18, device="cuda")
```

The packed container uses arbitrary-width exact integer IDs, independent template,
delta, and recipe indexes, atomic replacement, payload checksums, and memory-mapped
page reads. `DynamicTransformer.export_routing(directory)` exports every routed
pool plus a manifest.

See [docs/SPRC.md](docs/SPRC.md) for the complete storage/runtime design.

## RoPE

New-DT stores no learned or sinusoidal additive position table. Rotary position
embedding is applied directly to Q and K in every attention layer.

## Adam and structural updates

Recommended order:

```text
micro-batch forward/backward
capture exact route-slot gradient evidence
repeat until gradient accumulation completes
Adam step
periodic split/merge pass
route compaction when useful
zero gradients
```

Use `torch.optim.Adam`, or AdamW with `weight_decay=0`, for the current incremental
merge-value index.

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

## Small GPT versus sDT comparison

Train a conventional shared-matrix GPT and sDT on the same dataset-derived
word-space vocabulary and the exact same precomputed batches:

```bash
python compare_small.py \
  --data dataset.txt \
  --model both \
  --d-model 32 \
  --heads 4 \
  --layers 2 \
  --ffn-dim 128 \
  --seq-len 64 \
  --steps 500 \
  --batch-size 8
```

The installed command is also available:

```bash
new-dt-compare --data dataset.txt --model both ...
```

Both models use RMSNorm, RoPE, SwiGLU, causal attention, bias-free projections,
identical optimizer settings, and **untied** embedding/LM-head storage. The GPT
uses conventional shared Q/K/V/O and FFN matrices; sDT uses token-owned scalar
routes. Set `--structure-interval 0` for a static-routing ablation or use a positive
interval to enable split/merge.

See [docs/SMALL_COMPARISON.md](docs/SMALL_COMPARISON.md) for all CLI controls,
fairness rules, output files, and recommended starting configurations.

## Minimal usage

```python
import torch
from new_dt import (
    DynamicStructureController,
    DynamicTransformer,
    DynamicTransformerConfig,
)

config = DynamicTransformerConfig(
    vocab_size=32,
    d_model=8,
    n_heads=2,
    n_layers=1,
    ffn_dim=16,
    route_page_size=1024,
    route_templates_per_page=16,
    route_linear_out_tile=64,
    route_lm_head_tile=1024,
)
model = DynamicTransformer(config)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
controller = DynamicStructureController(structure_interval=100)

input_ids = torch.randint(0, config.vocab_size, (2, 12))
output = model(input_ids, labels=input_ids, collect_route_grads=True)
assert output.loss is not None
output.loss.backward()
controller.collect(model)
optimizer.step()
controller.maybe_restructure(model, optimizer, optimizer_step=1)
optimizer.zero_grad(set_to_none=True)

print(model.pool_summary())
print(model.routing_storage_summary())
```

## Implementation status

The repository implements exact SPRC split/merge semantics, adaptive selectors,
immutable templates, shared deltas, exceptions, compaction, exact reverse queries,
packed on-disk containers, mmap selective reads, route caches, tiled execution,
RoPE, storage telemetry, benchmark coverage, and a matched small-GPT comparison
runner. Native fused CUDA decoding and distributed route sharding remain optional
future backends; the current tiled PyTorch path is exact and works on CPU or GPU.
