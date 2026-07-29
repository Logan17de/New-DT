# New-DT: Token-Owned Scalar-Pool Dynamic Transformer

New-DT is a research reference implementation of a Transformer-shaped model in
which **every vocabulary token owns routes into separate scalar-neuron pools**.
Tokens may share individual scalars. Adam updates each unique scalar once using
the sum of every gradient contribution that reached it.

This repository implements the architecture discussed for the new DT branch. It
is deliberately explicit and small-scale: the goal is to make ownership,
sharing, Adam behavior, split, and merge inspectable before optimizing kernels.

## Architecture

```text
Input token IDs
    │
    ├── token-owned routes → Embedding scalar pool
    │
    ▼
Shared residual stream + fixed positional encoding
    │
    ├── Shared RMSNorm
    ├── token-owned Q/K/V/O routes → Attention scalar pools
    ├── shared causal mask + softmax
    └── shared residual addition
    │
    ├── Shared RMSNorm
    ├── token-owned Up/Gate/Down routes → FFN scalar pools
    └── shared residual addition
    │
    ▼
Shared final RMSNorm
    │
    └── candidate-token-owned routes → LM scalar pool
        ▼
      logits
```

### Pool separation

There are four conceptual families, and their storage never overlaps:

1. **Embedding pool**
2. **Attention family** — separate Q, K, V, and O pools for every layer
3. **FFN family** — separate Up, Gate, and Down pools for every layer
4. **LM pool** — owned by candidate output tokens

The implementation uses an even stricter boundary than one pool per family:
each layer/projection has its own `ScalarPool`, preventing accidental merges
between unrelated operations.

### What stays common

These components are shared because they are communication or stabilization,
not token-owned knowledge capacity:

- residual stream and residual addition;
- RMSNorm parameters;
- fixed positional encoding;
- causal mask and attention softmax;
- layer dimensions and topology.

## Scalar sharing example

With four route slots and `initial_shared_fraction=0.5`:

```text
Token A → [0, 1, 2, 3]
Token B → [0, 1, 4, 5]
```

Both tokens have four scalar neurons, share two, and together use six unique
scalars. When both appear, autograd adds their contributions to scalars `0` and
`1`; Adam updates six unique scalars rather than eight token occurrences.

## Adam and delayed structure changes

The intended training order is:

```text
micro-batch forward/backward
collect owner-specific route evidence
repeat until gradient accumulation completes
Adam step
periodic split/merge analysis
zero gradients
```

Adam remains ordinary Adam/AdamW. It stores one first moment and one second
moment for each scalar-pool entry. Routes—not Adam—decide where gradients go.

Pools are preallocated. A split therefore activates a free scalar slot rather
than replacing the PyTorch `Parameter`, preserving optimizer tensor shapes.
The new slot copies the source weight and, by default, its Adam moments.

## Split and merge

`DynamicStructureController` captures gradients on the materialized route tensor
before PyTorch sums them into a shared scalar. This preserves evidence such as:

```text
Token A → scalar 75 → positive gradient history
Token B → scalar 75 → negative gradient history
```

At a structural interval, persistent opposing owner gradients can trigger:

```text
Before: A → 75, B → 75
After:  A → 75', B → 75
```

Merge is conservative and local to one routed tensor. Two scalars are eligible
only when their values and gradient histories are close and their owner sets do
not overlap.

A forced final structural pass can be run at the end of training, so evidence
from a partial last interval is not discarded.

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Run the synthetic next-token demo:

```bash
python train_demo.py --steps 20 --grad-accum 2
```

## Minimal usage

```python
import torch
from new_dt import DynamicTransformer, DynamicTransformerConfig

config = DynamicTransformerConfig(
    vocab_size=32,
    d_model=8,
    n_heads=2,
    n_layers=1,
    ffn_dim=16,
    initial_shared_fraction=0.5,
)
model = DynamicTransformer(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

input_ids = torch.randint(0, config.vocab_size, (2, 12))
output = model(input_ids, labels=input_ids)
output.loss.backward()
optimizer.step()
```

## Important limitation

A token-specific full matrix contains many scalar routes. This reference model
therefore scales poorly in route-table memory and gathered tensors. That is not
hidden: it is the central engineering problem of strict scalar DT.

The next performance branch should preserve scalar ownership while testing:

- packed and deduplicated route tables;
- sparse gather/scatter kernels;
- route caching during inference;
- limited token-owned matrices or structured low-rank assembly;
- bounded pool growth and free-slot recycling;
- parameter-matched comparisons with dense Transformer, LoRA, and MOD.

The current repository establishes a correct, testable baseline before those
optimizations change the semantics.
