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
    → immutable base-template selector
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
- The same patch reused by multiple token pages becomes one shared delta.
- A large patch is resolved once and absorbed into a new immutable template.
- Merges remove or rewrite patches and may make old templates unused.

### Exact reverse ownership

SPRC does not store a dense second copy of every relation edge. Exact owners are
derived from:

```text
template → token users
template → scalar offsets
sparse deviations → exact route locations
```

This preserves exact split and merge while keeping the persistent representation
program-based.

### Selective decoding

`resolve_page(token_id, page_id)` reconstructs one independent page. Production
kernels can decode only the current layer/matrix tile rather than materializing a
complete token route.

See [docs/SPRC.md](docs/SPRC.md) for the storage model.

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

Run the synthetic demo:

```bash
python train_demo.py --steps 20 --grad-accum 2
```

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
```

## Reference implementation status

The repository now implements the exact SPRC semantics, immutable templates,
shared deltas, exceptions, compaction, program-derived reverse ownership, RoPE,
and split/merge integration. The Python implementation prioritizes correctness
and inspectability. Packed selector streams, on-disk page recipes, fused
unpack-and-gather CUDA kernels, tile caches, and distributed storage remain the
next performance layer.
