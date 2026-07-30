# Selective Page Reconstruction Compression (SPRC)

SPRC preserves one exact scalar-neuron ID per logical route slot without storing a
dense vocabulary-by-route integer tensor.

```text
token + page
    -> adaptive template selector
    -> immutable base template
    -> optional shared sparse delta
    -> rare token-specific exceptions
    -> exact resolved neuron IDs
```

Pages are decoding and storage units only. They are **not** the semantic sharing
unit: the same scalar may still occur at arbitrary offsets in unrelated pages.

## Compact selectors

The initial selector for a token is identical across route pages, so SPRC stores
that default once per token. Only page-specific deviations are retained.

```text
token default selector
    + sparse page overrides
    + dense page vector only when one page diverges heavily
```

This avoids the previous `vocab_size × num_pages × INT32` selector tensor. A dense
page can demote back to sparse storage after compaction restores similarity.

## Split, delta, and template promotion

A split clones the scalar value and Adam moments, then adds one exception:

```text
base template: offset 17 -> scalar 75
token exception: offset 17 -> scalar 901
```

The base template remains immutable. When the same **full patch relative to the
base template** appears repeatedly, SPRC interns it as one shared delta. When a
patch becomes large, the resolved page is promoted into a new immutable template.
Existing users of the old template are untouched.

Using the full patch matters: a token can already use a shared delta and later add
new exceptions without losing the earlier delta during promotion.

## Exact merge ownership

The exact owner set is derived from:

- scalar -> relevant template/page/offset references;
- template users derived lazily from compact selectors;
- sparse locations that override their base template.

A merge redirects only exact owners. Template-user results are cached rather than
stored as a permanent second copy of every selector edge.

## Runtime page cache

Immutable `template + delta` pages are cached by:

```text
(device, page ID, template ID, delta ID)
```

Token-specific exceptions are applied after the shared cached page is copied.
`resolve_page_batch(...)` groups tokens by route program, so one shared page is
decoded once and broadcast to every matching token.

## Tiled execution

The model does not expand a complete token-owned attention or FFN matrix at once.
`TokenRoutedLinear` resolves output-row tiles:

```text
route slice -> scalar gather -> small weight tile -> matrix multiply -> next tile
```

The LM head is similarly reconstructed in vocabulary tiles. This bounds peak route
ID and gathered-weight memory while preserving the exact same output and exact
route-slot gradient evidence.

Configuration controls:

- `route_linear_out_tile`
- `route_lm_head_tile`
- `route_materialize_token_chunk`
- `route_cache_pages`

## Packed container and selective disk reads

`RoutedParameterTensor.export_packed(path)` writes an atomic `.sprc` container:

- arbitrary-width packed neuron IDs;
- packed per-token selector defaults;
- sparse page selector overrides;
- independently indexed immutable templates;
- independently indexed shared deltas;
- independently indexed token-page recipes;
- payload checksum.

`PackedSPRCReader` memory-maps the file and reconstructs only the requested page or
route slice. Templates, deltas, selector overrides, and recipes are loaded lazily
and cached. The operating system can satisfy access through range-backed mmap pages
without decompressing the complete routing store.

```python
routed.export_packed("embedding.sprc")

from new_dt import PackedSPRCReader
with PackedSPRCReader("embedding.sprc") as reader:
    route_page = reader.resolve_page(token_id=732, page_id=18, device="cuda")
```

`DynamicTransformer.export_routing(directory)` writes one independent container per
routed tensor plus a manifest.

## Storage and cache telemetry

- `routing_storage_estimate()` estimates packed payload and sparse index costs.
- `route_cache_stats()` reports cache hits and misses.
- `DynamicTransformer.routing_storage_summary()` aggregates every routed tensor.
- `DynamicTransformer.route_cache_summary()` exposes runtime cache behavior.

## RoPE

New-DT stores no additive sinusoidal or learned position vector. Rotary position
embedding is applied directly to Q and K, and its cosine/sine tensors are cached by
sequence length, device, and dtype.
