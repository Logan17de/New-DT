# New-DT: Token-Owned Scalar-Pool Dynamic Transformer

New-DT is a research reference implementation of a Transformer-shaped model in
which every vocabulary token owns routes into separate scalar-neuron pools.
Tokens may reuse the same scalar from **any position** inside a vector or matrix.
Adam updates each unique scalar once using the combined gradient that reached it.

The current goal is correctness and inspectability before custom sparse kernels.

## Architecture

```text
Input token IDs
    │
    ├── token-owned routes → Embedding scalar pool
    ▼
Shared residual stream + positional encoding
    │
    ├── Shared RMSNorm
    ├── token-owned Q/K/V/O routes → separate attention pools
    ├── shared causal mask + softmax
    └── shared residual addition
    │
    ├── Shared RMSNorm
    ├── token-owned Up/Gate/Down routes → separate FFN pools
    └── shared residual addition
    ▼
Shared final RMSNorm
    │
    └── candidate-token-owned routes → LM scalar pool
        ▼
      logits
```

Pool storage never overlaps across embedding, Q, K, V, O, FFN projections, or
LM output. Residuals, normalization, masking, softmax, positional encoding, and
topology remain common so all tokens communicate in one representation space.

## Global scalar sharing inside a pool

A scalar is not tied to one vector coordinate:

```text
scalar 75
├── Token A, route slot 3
├── Token B, route slot 41
└── Token C, route slot 92
```

If their route gradients are:

```text
A/3  → +0.40
B/41 → +0.30
C/92 → -0.50
```

Autograd gives Adam one pool gradient:

```text
gradient[75] = +0.20
```

New-DT also retains the three exact route contributions. Persistent conflict can
therefore split only `C/92`:

```text
A/3  → 75
B/41 → 75
C/92 → 75'
```

## Forward and reverse routing indexes

Every routed tensor maintains both directions:

```text
(token, route slot) → scalar
scalar → all (token, route slot) locations
```

The reverse index also maintains each scalar's total usage count. Split and merge
operations no longer search the full route table.

- `reroute_slot(...)` changes one exact route cell.
- `replace_scalar_everywhere(...)` visits only locations already indexed for the
  source scalar.
- The reverse index is rebuilt automatically after checkpoint loading.

## Adam and gradient accumulation

Recommended order:

```text
micro-batch forward/backward
capture route-slot gradient evidence
repeat until accumulation completes
Adam step
periodic affected-neuron split/merge pass
zero gradients
```

Adam stores one first moment and second moment for each scalar-pool entry. Routes,
not Adam, determine where gradients go. A split copies the source value and its
Adam moments into a preallocated free slot.

For incremental merge indexing, use `torch.optim.Adam`, or AdamW with
`weight_decay=0`. Nonzero global/decoupled weight decay changes untouched entries
inside the whole pool tensor and invalidates an affected-only value index.

## Split threshold

Conflict evidence is stored by:

```text
(pool, token ID, exact route slot, scalar ID)
```

Only scalars affected since the previous structural pass are analyzed. A scalar
used in many places receives a higher effective split threshold:

```text
threshold = base_threshold
          + owner_threshold_scale × log2(max(1, owner_count / 2))
```

The threshold is capped by `max_conflict_threshold`. This makes heavily shared
neurons harder—but not impossible—to split.

## Incremental merge

The controller maintains scalar-value buckets per pool. After Adam, only affected
or still-momentum-active scalars refresh their bucket membership.

An affected scalar is compared only with nearby buckets, then merged only when:

- scalar values are within `merge_weight_tolerance`;
- gradient histories are within `merge_gradient_tolerance`;
- both have enough gradient samples.

The less-used scalar is redirected into the more-used scalar, minimizing route
map writes. Scalars created by a split are protected from immediate re-merging in
the same structural pass.

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

The tests cover:

- forward/backward and Adam;
- strict pool separation and shared RMSNorm;
- one scalar reused at different route slots;
- exact-slot splitting and Adam-state copying;
- opposing route gradients that cancel in the pool gradient;
- conflict-based controller splitting;
- affected-neighborhood merging through the reverse map.

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
    initial_shared_fraction=0.5,
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

## Current limitation

A token-specific full matrix contains many scalar routes, so route-table memory
and gathered tensors still scale poorly. The next performance work should focus
on packed route storage, sparse gather/scatter kernels, inference route caching,
and bounded pool growth without changing exact scalar ownership semantics.
