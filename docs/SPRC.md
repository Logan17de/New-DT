# Selective Page Reconstruction Compression (SPRC)

SPRC is the persistent route engine for New-DT. It preserves one exact scalar
neuron ID per logical route slot without storing a dense vocabulary-by-route
integer tensor.

```text
token + page
    -> immutable base-template selector
    -> optional shared sparse delta
    -> rare token-specific exceptions
    -> exact resolved neuron IDs
```

Pages are decoding and storage units only. They are **not** the semantic sharing
unit: the same scalar may still occur at arbitrary offsets in unrelated pages.

## Split

A split clones the scalar value and Adam moments, then adds one exception:

```text
base template: offset 17 -> scalar 75
token exception: offset 17 -> scalar 901
```

The base template remains immutable.

## Delta promotion

If the same sparse exception pattern occurs on multiple token pages, SPRC interns
it once as a shared delta and removes the repeated token exceptions.

## Template promotion

When a page accumulates a large delta, SPRC resolves the page once, interns the
result as a new immutable template, changes that token's selector, and removes the
old delta/exception chain. Existing users of the old template are untouched.

## Merge

The exact owner set is derived from:

- template -> token users;
- template -> scalar offsets;
- sparse locations that override their base template.

A merge redirects only those exact owners. Resulting repeated changes may become
a shared delta or a new template during compaction.

## Selective decoding

`resolve_page(token_id, page_id)` reconstructs one page independently. Runtime
kernels can therefore decode only the current layer/matrix tile rather than the
complete token route.

## Storage estimate

`RoutedParameterTensor.routing_storage_estimate()` reports packed payload bits for
selectors, immutable templates, shared deltas, and unique exceptions. The Python
reference uses dictionaries and tensors for inspectability; it does not claim the
Python object overhead equals the packed production format.

## RoPE

New-DT no longer adds a sinusoidal or learned position vector to embeddings.
Rotary position embedding is applied directly to Q and K inside every attention
layer.
