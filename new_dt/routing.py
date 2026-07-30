from __future__ import annotations

import hashlib
import math
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn

from .selectors import AdaptiveSelectorTable


@dataclass(frozen=True, slots=True)
class RoutePageRecipe:
    """Inspectable recipe for one token/page without materializing the full route."""

    token_id: int
    page_id: int
    template_id: int
    delta_id: int | None
    exception_count: int


class SelectivePageReconstructionStore(nn.Module):
    """Exact immutable-template route storage with sparse deltas and exceptions.

    The persistent selector layer is itself hierarchical: one compact default is
    stored per token, and only page-specific selector divergence is retained.
    Decoded base+delta pages are cached by immutable program identity. Token-only
    exceptions are applied after the shared page is fetched from that cache.
    """

    def __init__(
        self,
        vocab_size: int,
        route_size: int,
        *,
        page_size: int,
        templates_per_page: int,
        shared_fraction: float,
        delta_promotion_threshold: int,
        template_promotion_threshold: int,
        template_promotion_fraction: float,
        shared_delta_min_reuse: int,
        cache_pages: int = 256,
        selector_dense_promotion_fraction: float = 0.35,
        selector_dense_demotion_fraction: float = 0.15,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or route_size <= 0:
            raise ValueError("vocab_size and route_size must be positive")
        if page_size <= 0 or templates_per_page <= 0:
            raise ValueError("page_size and templates_per_page must be positive")
        if cache_pages < 0:
            raise ValueError("cache_pages must be non-negative")

        self.vocab_size = vocab_size
        self.route_size = route_size
        self.page_size = page_size
        self.num_pages = math.ceil(route_size / page_size)
        self.templates_per_page = min(templates_per_page, vocab_size)
        self.shared_fraction = shared_fraction
        self.delta_promotion_threshold = delta_promotion_threshold
        self.template_promotion_threshold = template_promotion_threshold
        self.template_promotion_fraction = template_promotion_fraction
        self.shared_delta_min_reuse = shared_delta_min_reuse
        self.cache_pages = cache_pages

        self._selectors = AdaptiveSelectorTable(
            vocab_size,
            self.num_pages,
            self.templates_per_page,
            dense_promotion_fraction=selector_dense_promotion_fraction,
            dense_demotion_fraction=selector_dense_demotion_fraction,
        )

        self._templates: list[list[Tensor]] = []
        self._delta_banks: list[list[tuple[tuple[int, int], ...]]] = [
            [] for _ in range(self.num_pages)
        ]
        self._delta_selectors: dict[int, int] = {}
        self._exceptions: dict[int, dict[int, int]] = {}

        # Fast interning avoids linear scans over every existing route program.
        self._template_hash_to_ids: list[dict[bytes, list[int]]] = [
            defaultdict(list) for _ in range(self.num_pages)
        ]
        self._delta_lookup: list[dict[tuple[tuple[int, int], ...], int]] = [
            {} for _ in range(self.num_pages)
        ]

        # Exact reverse representation. Template ownership is derived lazily from
        # selectors; only scalar offsets and sparse deviations are permanently held.
        self._template_scalar_offsets: list[list[dict[int, tuple[int, ...]]]] = []
        self._scalar_template_refs: dict[
            int, list[tuple[int, int, tuple[int, ...]]]
        ] = defaultdict(list)
        self._explicit_locations: set[int] = set()
        self._page_explicit_locations: dict[int, set[int]] = {}
        self._explicit_to_scalar: dict[int, int] = {}
        self._patch_to_locations: dict[int, set[int]] = defaultdict(set)
        self._usage_counts: Counter[int] = Counter()

        # Full page-patch signatures make repeated-delta discovery O(reuse) rather
        # than rescanning every token page after each split.
        self._page_patch_signature: dict[int, tuple[tuple[int, int], ...]] = {}
        self._patch_signature_to_instances: dict[
            tuple[int, tuple[tuple[int, int], ...]], set[int]
        ] = defaultdict(set)

        self._program_page_cache: OrderedDict[
            tuple[str, int, int, int | None], Tensor
        ] = OrderedDict()
        self._template_user_cache: OrderedDict[
            tuple[int, int], tuple[int, ...]
        ] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        self.required_pool_size = self._initialize_templates()
        self._rebuild_intern_lookups()
        self.rebuild_indexes()

    def _page_length(self, page_id: int) -> int:
        start = page_id * self.page_size
        return min(self.page_size, self.route_size - start)

    def _page_instance(self, token_id: int, page_id: int) -> int:
        return token_id * self.num_pages + page_id

    def _split_page_instance(self, page_instance: int) -> tuple[int, int]:
        return divmod(page_instance, self.num_pages)

    def _pack_location(self, token_id: int, route_slot: int) -> int:
        return token_id * self.route_size + route_slot

    def _unpack_location(self, packed: int) -> tuple[int, int]:
        return divmod(packed, self.route_size)

    def _initialize_templates(self) -> int:
        """Create route programs without a dense vocabulary-by-route tensor."""

        cursor = self.route_size
        for page_id in range(self.num_pages):
            start = page_id * self.page_size
            length = self._page_length(page_id)
            base = torch.arange(start, start + length, dtype=torch.long)
            shared_count = min(length, max(0, int(round(length * self.shared_fraction))))
            page_templates = [base]
            for template_id in range(1, self.templates_per_page):
                template = torch.empty(length, dtype=torch.long)
                if shared_count:
                    shift = template_id % max(length, 1)
                    shared_source = torch.roll(base, shifts=shift)
                    template[:shared_count] = shared_source[:shared_count]
                private_count = length - shared_count
                if private_count:
                    template[shared_count:] = torch.arange(
                        cursor, cursor + private_count, dtype=torch.long
                    )
                    cursor += private_count
                page_templates.append(template)
            self._templates.append(page_templates)
        return cursor

    @staticmethod
    def _template_hash(route: Tensor) -> bytes:
        route_cpu = route.detach().to(dtype=torch.long, device="cpu").contiguous()
        return hashlib.blake2b(route_cpu.numpy().tobytes(), digest_size=16).digest()

    def _rebuild_intern_lookups(self) -> None:
        self._template_hash_to_ids = [defaultdict(list) for _ in range(self.num_pages)]
        for page_id, templates in enumerate(self._templates):
            for template_id, template in enumerate(templates):
                self._template_hash_to_ids[page_id][self._template_hash(template)].append(
                    template_id
                )
        self._delta_lookup = [
            {delta: delta_id for delta_id, delta in enumerate(bank)}
            for bank in self._delta_banks
        ]

    def _template_id(self, token_id: int, page_id: int) -> int:
        return self._selectors.get(token_id, page_id)

    def _base_page(self, token_id: int, page_id: int) -> Tensor:
        return self._templates[page_id][self._template_id(token_id, page_id)]

    def _delta_id(self, token_id: int, page_id: int) -> int | None:
        return self._delta_selectors.get(self._page_instance(token_id, page_id))

    def _delta(self, token_id: int, page_id: int) -> tuple[tuple[int, int], ...]:
        delta_id = self._delta_id(token_id, page_id)
        if delta_id is None:
            return ()
        return self._delta_banks[page_id][delta_id]

    def _remember_cache(self, cache: OrderedDict, key: object, value: Tensor | tuple[int, ...]):
        if not self.cache_pages:
            return value
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.cache_pages:
            cache.popitem(last=False)
        return value

    def clear_cache(self) -> None:
        self._program_page_cache.clear()
        self._template_user_cache.clear()

    def cache_stats(self) -> dict[str, int]:
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "program_pages": len(self._program_page_cache),
            "template_user_entries": len(self._template_user_cache),
        }

    def _program_page(
        self,
        token_id: int,
        page_id: int,
        device: torch.device | str | None,
    ) -> Tensor:
        device_obj = torch.device("cpu" if device is None else device)
        template_id = self._template_id(token_id, page_id)
        delta_id = self._delta_id(token_id, page_id)
        key = (str(device_obj), page_id, template_id, delta_id)
        cached = self._program_page_cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            self._program_page_cache.move_to_end(key)
            return cached
        self._cache_misses += 1
        route = self._templates[page_id][template_id].to(
            device=device_obj, dtype=torch.long, non_blocking=True
        ).clone()
        if delta_id is not None:
            delta = self._delta_banks[page_id][delta_id]
            if delta:
                offsets = torch.tensor([item[0] for item in delta], device=device_obj)
                values = torch.tensor([item[1] for item in delta], device=device_obj)
                route[offsets] = values
        return self._remember_cache(self._program_page_cache, key, route)  # type: ignore[return-value]

    def resolve_page(
        self,
        token_id: int,
        page_id: int,
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Decode one page independently, using the immutable-program cache."""

        if not 0 <= token_id < self.vocab_size:
            raise IndexError("token_id outside vocabulary")
        if not 0 <= page_id < self.num_pages:
            raise IndexError("page_id outside route")
        route = self._program_page(token_id, page_id, device).clone()
        exceptions = self._exceptions.get(self._page_instance(token_id, page_id))
        if exceptions:
            device_obj = route.device
            offsets = torch.tensor(list(exceptions), device=device_obj)
            values = torch.tensor(list(exceptions.values()), device=device_obj)
            route[offsets] = values
        return route

    def resolve_page_batch(
        self,
        token_ids: Tensor,
        page_id: int,
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Resolve a page for many tokens, decoding each shared recipe only once."""

        if not 0 <= page_id < self.num_pages:
            raise IndexError("page_id outside route")
        flat_tokens = [int(value) for value in token_ids.detach().reshape(-1).cpu().tolist()]
        device_obj = torch.device(token_ids.device if device is None else device)
        page_length = self._page_length(page_id)
        result = torch.empty(
            len(flat_tokens), page_length, dtype=torch.long, device=device_obj
        )
        groups: dict[tuple[int, int | None], list[tuple[int, int]]] = defaultdict(list)
        for row, token_id in enumerate(flat_tokens):
            if not 0 <= token_id < self.vocab_size:
                raise IndexError("token_id outside vocabulary")
            groups[(self._template_id(token_id, page_id), self._delta_id(token_id, page_id))].append(
                (row, token_id)
            )

        for members in groups.values():
            first_token = members[0][1]
            shared = self._program_page(first_token, page_id, device_obj)
            rows = torch.tensor([row for row, _ in members], device=device_obj)
            result[rows] = shared
            for row, token_id in members:
                exceptions = self._exceptions.get(self._page_instance(token_id, page_id))
                if not exceptions:
                    continue
                offsets = torch.tensor(list(exceptions), device=device_obj)
                values = torch.tensor(list(exceptions.values()), device=device_obj)
                result[row, offsets] = values
        return result.view(*token_ids.shape, page_length)

    def resolve_slice(
        self,
        token_ids: Tensor,
        start: int,
        stop: int,
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Resolve only a contiguous route interval, touching intersecting pages."""

        if not 0 <= start <= stop <= self.route_size:
            raise IndexError("invalid route slice")
        target_device = token_ids.device if device is None else torch.device(device)
        if start == stop:
            return torch.empty(*token_ids.shape, 0, dtype=torch.long, device=target_device)
        first_page = start // self.page_size
        last_page = (stop - 1) // self.page_size
        parts: list[Tensor] = []
        for page_id in range(first_page, last_page + 1):
            page = self.resolve_page_batch(token_ids, page_id, device=target_device)
            page_start = page_id * self.page_size
            local_start = max(start, page_start) - page_start
            local_stop = min(stop, page_start + self._page_length(page_id)) - page_start
            parts.append(page[..., local_start:local_stop])
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def resolve_token(
        self, token_id: int, *, device: torch.device | str | None = None
    ) -> Tensor:
        ids = torch.tensor([token_id], dtype=torch.long)
        return self.resolve_slice(ids, 0, self.route_size, device=device)[0]

    def resolve(self, token_ids: Tensor, *, device: torch.device | str | None = None) -> Tensor:
        return self.resolve_slice(token_ids, 0, self.route_size, device=device)

    def materialize_all(
        self,
        *,
        device: torch.device | str | None = None,
        token_chunk_size: int = 128,
    ) -> Tensor:
        if token_chunk_size <= 0:
            raise ValueError("token_chunk_size must be positive")
        rows: list[Tensor] = []
        for start in range(0, self.vocab_size, token_chunk_size):
            token_ids = torch.arange(start, min(self.vocab_size, start + token_chunk_size))
            rows.append(self.resolve(token_ids, device=device))
        return torch.cat(rows, dim=0)

    def scalar_at(self, token_id: int, route_slot: int) -> int:
        if not 0 <= token_id < self.vocab_size:
            raise IndexError("token_id outside vocabulary")
        if not 0 <= route_slot < self.route_size:
            raise IndexError("route_slot outside parameter tensor")
        page_id, offset = divmod(route_slot, self.page_size)
        page_instance = self._page_instance(token_id, page_id)
        exception = self._exceptions.get(page_instance, {}).get(offset)
        if exception is not None:
            return int(exception)
        for delta_offset, scalar in self._delta(token_id, page_id):
            if delta_offset == offset:
                return int(scalar)
        return int(self._base_page(token_id, page_id)[offset].item())

    def page_recipe(self, token_id: int, page_id: int) -> RoutePageRecipe:
        page_instance = self._page_instance(token_id, page_id)
        return RoutePageRecipe(
            token_id=token_id,
            page_id=page_id,
            template_id=self._template_id(token_id, page_id),
            delta_id=self._delta_selectors.get(page_instance),
            exception_count=len(self._exceptions.get(page_instance, {})),
        )

    def selector_state(self) -> dict[str, object]:
        return self._selectors.state()

    def selector_override_pages(self) -> tuple[int, ...]:
        return self._selectors.override_pages()

    def selector_page_overrides(self, page_id: int) -> dict[int, int]:
        return self._selectors.page_overrides(page_id)

    def max_template_id(self) -> int:
        return max(
            self._selectors.max_template_id(),
            max((len(page) - 1 for page in self._templates), default=0),
        )

    def _build_template_indexes(self) -> None:
        self._template_scalar_offsets = []
        self._scalar_template_refs.clear()
        for page_id, templates in enumerate(self._templates):
            indexed_templates: list[dict[int, tuple[int, ...]]] = []
            for template_id, template in enumerate(templates):
                offsets: dict[int, list[int]] = defaultdict(list)
                for offset, scalar in enumerate(template.tolist()):
                    offsets[int(scalar)].append(offset)
                indexed = {scalar: tuple(items) for scalar, items in offsets.items()}
                indexed_templates.append(indexed)
                for scalar, scalar_offsets in indexed.items():
                    self._scalar_template_refs[scalar].append(
                        (page_id, template_id, scalar_offsets)
                    )
            self._template_scalar_offsets.append(indexed_templates)

    def _template_users(self, page_id: int, template_id: int) -> tuple[int, ...]:
        key = (page_id, template_id)
        cached = self._template_user_cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            self._template_user_cache.move_to_end(key)
            return cached
        self._cache_misses += 1
        selectors = self._selectors.values_for_page(page_id)
        users = tuple(
            int(token)
            for token in (selectors == template_id).nonzero(as_tuple=False).flatten().tolist()
        )
        return self._remember_cache(self._template_user_cache, key, users)  # type: ignore[return-value]

    def rebuild_indexes(self) -> None:
        """Rebuild compact reverse and patch indexes after checkpoint loading."""

        self.clear_cache()
        self._build_template_indexes()
        self._usage_counts.clear()
        self._explicit_locations.clear()
        self._page_explicit_locations.clear()
        self._explicit_to_scalar.clear()
        self._patch_to_locations.clear()
        self._page_patch_signature.clear()
        self._patch_signature_to_instances.clear()

        for page_id, templates in enumerate(self._templates):
            template_counts = self._selectors.template_counts(page_id)
            for template_id, user_count in template_counts.items():
                if not user_count or template_id >= len(templates):
                    continue
                template = templates[template_id]
                for scalar, count in Counter(int(v) for v in template.tolist()).items():
                    self._usage_counts[scalar] += user_count * count

        touched_pages = set(self._delta_selectors) | set(self._exceptions)
        for page_instance in touched_pages:
            token_id, page_id = self._split_page_instance(page_instance)
            base = self._base_page(token_id, page_id)
            resolved = self.resolve_page(token_id, page_id)
            for offset, (base_scalar, final_scalar) in enumerate(
                zip(base.tolist(), resolved.tolist(), strict=True)
            ):
                if base_scalar == final_scalar:
                    continue
                self._usage_counts[int(base_scalar)] -= 1
                self._usage_counts[int(final_scalar)] += 1
                self._add_explicit(token_id, page_id, offset, int(final_scalar))
            self._update_patch_signature(token_id, page_id, base=base, resolved=resolved)

    def _add_explicit(self, token_id: int, page_id: int, offset: int, scalar: int) -> None:
        route_slot = page_id * self.page_size + offset
        packed = self._pack_location(token_id, route_slot)
        page_instance = self._page_instance(token_id, page_id)
        self._explicit_locations.add(packed)
        self._page_explicit_locations.setdefault(page_instance, set()).add(packed)
        self._explicit_to_scalar[packed] = scalar
        self._patch_to_locations[scalar].add(packed)

    def _remove_explicit(self, packed: int) -> None:
        scalar = self._explicit_to_scalar.pop(packed, None)
        if scalar is None:
            return
        self._explicit_locations.discard(packed)
        self._patch_to_locations[scalar].discard(packed)
        if not self._patch_to_locations[scalar]:
            self._patch_to_locations.pop(scalar, None)
        token_id, route_slot = self._unpack_location(packed)
        page_id = route_slot // self.page_size
        page_instance = self._page_instance(token_id, page_id)
        locations = self._page_explicit_locations.get(page_instance)
        if locations is not None:
            locations.discard(packed)
            if not locations:
                self._page_explicit_locations.pop(page_instance, None)

    def _remove_patch_signature(self, page_instance: int) -> None:
        old = self._page_patch_signature.pop(page_instance, None)
        if old is None:
            return
        _, page_id = self._split_page_instance(page_instance)
        key = (page_id, old)
        instances = self._patch_signature_to_instances.get(key)
        if instances is not None:
            instances.discard(page_instance)
            if not instances:
                self._patch_signature_to_instances.pop(key, None)

    def _update_patch_signature(
        self,
        token_id: int,
        page_id: int,
        *,
        base: Tensor | None = None,
        resolved: Tensor | None = None,
    ) -> tuple[tuple[int, int], ...]:
        page_instance = self._page_instance(token_id, page_id)
        self._remove_patch_signature(page_instance)
        base = self._base_page(token_id, page_id) if base is None else base
        resolved = self.resolve_page(token_id, page_id) if resolved is None else resolved
        patch = tuple(
            (offset, int(final))
            for offset, (original, final) in enumerate(
                zip(base.tolist(), resolved.tolist(), strict=True)
            )
            if original != final
        )
        if patch:
            self._page_patch_signature[page_instance] = patch
            self._patch_signature_to_instances[(page_id, patch)].add(page_instance)
        return patch

    def _refresh_page_explicit_index(self, token_id: int, page_id: int) -> None:
        page_instance = self._page_instance(token_id, page_id)
        for packed in tuple(self._page_explicit_locations.get(page_instance, ())):
            self._remove_explicit(packed)
        base = self._base_page(token_id, page_id)
        resolved = self.resolve_page(token_id, page_id)
        for offset, (base_scalar, final_scalar) in enumerate(
            zip(base.tolist(), resolved.tolist(), strict=True)
        ):
            if base_scalar != final_scalar:
                self._add_explicit(token_id, page_id, offset, int(final_scalar))
        self._update_patch_signature(token_id, page_id, base=base, resolved=resolved)

    def _replace_exceptions(self, page_instance: int, values: dict[int, int]) -> None:
        if values:
            self._exceptions[page_instance] = values
        else:
            self._exceptions.pop(page_instance, None)

    def reroute_slot(
        self,
        token_id: int,
        route_slot: int,
        new_scalar: int,
        *,
        expected_old_scalar: int | None = None,
        compact: bool = True,
    ) -> bool:
        """Change one exact route cell by writing/removing a sparse exception."""

        old_scalar = self.scalar_at(token_id, route_slot)
        if expected_old_scalar is not None and old_scalar != expected_old_scalar:
            return False
        if old_scalar == new_scalar:
            return False

        page_id, offset = divmod(route_slot, self.page_size)
        page_instance = self._page_instance(token_id, page_id)
        underlying = int(self._base_page(token_id, page_id)[offset].item())
        for delta_offset, scalar in self._delta(token_id, page_id):
            if delta_offset == offset:
                underlying = int(scalar)
                break

        exceptions = dict(self._exceptions.get(page_instance, {}))
        if new_scalar == underlying:
            exceptions.pop(offset, None)
        else:
            exceptions[offset] = int(new_scalar)
        self._replace_exceptions(page_instance, exceptions)

        self._usage_counts[old_scalar] -= 1
        self._usage_counts[new_scalar] += 1
        packed = self._pack_location(token_id, route_slot)
        self._remove_explicit(packed)
        base_scalar = int(self._base_page(token_id, page_id)[offset].item())
        if new_scalar != base_scalar:
            self._add_explicit(token_id, page_id, offset, int(new_scalar))
        self._update_patch_signature(token_id, page_id)

        if compact:
            self.compact_page(token_id, page_id)
        return True

    def _intern_delta(self, page_id: int, delta: tuple[tuple[int, int], ...]) -> int:
        existing = self._delta_lookup[page_id].get(delta)
        if existing is not None:
            return existing
        bank = self._delta_banks[page_id]
        bank.append(delta)
        delta_id = len(bank) - 1
        self._delta_lookup[page_id][delta] = delta_id
        return delta_id

    def _intern_template(self, page_id: int, route: Tensor) -> int:
        route_cpu = route.detach().to(dtype=torch.long, device="cpu").contiguous()
        digest = self._template_hash(route_cpu)
        for template_id in self._template_hash_to_ids[page_id].get(digest, ()):
            if torch.equal(self._templates[page_id][template_id], route_cpu):
                return template_id
        template_id = len(self._templates[page_id])
        self._templates[page_id].append(route_cpu)
        self._template_hash_to_ids[page_id][digest].append(template_id)
        offsets: dict[int, list[int]] = defaultdict(list)
        for offset, scalar in enumerate(route_cpu.tolist()):
            offsets[int(scalar)].append(offset)
        indexed = {scalar: tuple(items) for scalar, items in offsets.items()}
        self._template_scalar_offsets[page_id].append(indexed)
        for scalar, scalar_offsets in indexed.items():
            self._scalar_template_refs[scalar].append(
                (page_id, template_id, scalar_offsets)
            )
        return template_id

    def _set_template_selector(self, token_id: int, page_id: int, template_id: int) -> None:
        old_template = self._template_id(token_id, page_id)
        if not self._selectors.set(token_id, page_id, template_id):
            return
        self._template_user_cache.pop((page_id, old_template), None)
        self._template_user_cache.pop((page_id, template_id), None)

    def compact_page(self, token_id: int, page_id: int, *, force: bool = False) -> str | None:
        """Promote repeated full patches or absorb large patches into templates."""

        page_instance = self._page_instance(token_id, page_id)
        patch = self._update_patch_signature(token_id, page_id)
        if not patch:
            self._delta_selectors.pop(page_instance, None)
            self._exceptions.pop(page_instance, None)
            self._selectors.compact_page(page_id)
            return None

        page_length = self._page_length(page_id)
        large_patch = (
            len(patch) >= self.template_promotion_threshold
            or len(patch) / page_length >= self.template_promotion_fraction
        )
        if force or large_patch:
            resolved = self.resolve_page(token_id, page_id)
            template_id = self._intern_template(page_id, resolved)
            self._set_template_selector(token_id, page_id, template_id)
            self._delta_selectors.pop(page_instance, None)
            self._exceptions.pop(page_instance, None)
            self._refresh_page_explicit_index(token_id, page_id)
            self._selectors.compact_page(page_id)
            return "template"

        if len(patch) < self.delta_promotion_threshold:
            return None
        matching_instances = tuple(
            self._patch_signature_to_instances.get((page_id, patch), ())
        )
        if len(matching_instances) < self.shared_delta_min_reuse:
            return None

        delta_id = self._intern_delta(page_id, patch)
        for candidate in matching_instances:
            candidate_token, candidate_page = self._split_page_instance(candidate)
            self._delta_selectors[candidate] = delta_id
            self._exceptions.pop(candidate, None)
            self._refresh_page_explicit_index(candidate_token, candidate_page)
        return "delta"

    def compact_all(self, *, force: bool = False) -> dict[str, int]:
        counts = {"template": 0, "delta": 0}
        touched = tuple(set(self._delta_selectors) | set(self._exceptions))
        for page_instance in touched:
            token_id, page_id = self._split_page_instance(page_instance)
            result = self.compact_page(token_id, page_id, force=force)
            if result is not None:
                counts[result] += 1
        return counts

    def route_locations(self, scalar: int) -> tuple[tuple[int, int], ...]:
        """Enumerate exact owners from relevant templates plus deviations only."""

        locations = set(self._patch_to_locations.get(scalar, ()))
        for page_id, template_id, offsets in self._scalar_template_refs.get(scalar, ()):
            for token_id in self._template_users(page_id, template_id):
                for offset in offsets:
                    route_slot = page_id * self.page_size + offset
                    packed = self._pack_location(token_id, route_slot)
                    if packed not in self._explicit_locations:
                        locations.add(packed)
        return tuple(sorted(self._unpack_location(packed) for packed in locations))

    def replace_scalar_everywhere(self, old_scalar: int, new_scalar: int) -> int:
        locations = self.route_locations(old_scalar)
        changed = 0
        touched_pages: set[tuple[int, int]] = set()
        for token_id, route_slot in locations:
            if self.reroute_slot(
                token_id,
                route_slot,
                new_scalar,
                expected_old_scalar=old_scalar,
                compact=False,
            ):
                changed += 1
                touched_pages.add((token_id, route_slot // self.page_size))
        for token_id, page_id in touched_pages:
            self.compact_page(token_id, page_id)
        return changed

    def usage_count(self, scalar: int) -> int:
        return max(0, int(self._usage_counts.get(scalar, 0)))

    def iter_used_scalars(self) -> Iterable[int]:
        return (scalar for scalar, count in self._usage_counts.items() if count > 0)

    def storage_estimate(self, *, pool_size: int) -> dict[str, int]:
        """Estimate packed production payload, including sparse index keys."""

        neuron_bits = max(1, math.ceil(math.log2(max(pool_size, 2))))
        selector = self._selectors.packed_storage_bits()
        offset_bits = max(1, math.ceil(math.log2(max(self.page_size, 2))))
        page_instance_bits = max(
            1, math.ceil(math.log2(max(2, self.vocab_size * self.num_pages)))
        )

        template_values = sum(
            template.numel() for page in self._templates for template in page
        )
        template_payload = template_values * neuron_bits
        delta_entries = sum(len(delta) for bank in self._delta_banks for delta in bank)
        delta_payload = delta_entries * (offset_bits + neuron_bits)
        max_delta_count = max((len(bank) for bank in self._delta_banks), default=1)
        delta_id_bits = max(1, math.ceil(math.log2(max(2, max_delta_count))))
        delta_selector_payload = len(self._delta_selectors) * (
            page_instance_bits + delta_id_bits
        )
        exception_entries = sum(len(items) for items in self._exceptions.values())
        exception_payload = exception_entries * (offset_bits + neuron_bits)
        exception_page_index = len(self._exceptions) * page_instance_bits
        metadata_bits = 128 * (
            sum(len(page) for page in self._templates)
            + sum(len(bank) for bank in self._delta_banks)
            + len(set(self._delta_selectors) | set(self._exceptions))
        )
        total_bits = (
            selector["selector_bits"]
            + template_payload
            + delta_payload
            + delta_selector_payload
            + exception_payload
            + exception_page_index
            + metadata_bits
        )
        return {
            **selector,
            "template_bits": template_payload,
            "delta_bits": delta_payload + delta_selector_payload,
            "exception_bits": exception_payload + exception_page_index,
            "metadata_bits": metadata_bits,
            "total_bits": total_bits,
            "total_bytes": math.ceil(total_bits / 8),
            "neuron_id_bits": neuron_bits,
        }

    def export_packed(
        self,
        path: str | Path,
        *,
        pool_size: int,
    ) -> dict[str, int]:
        from .packed import PackedSPRCWriter

        return PackedSPRCWriter.write(self, path, pool_size=pool_size)

    def get_extra_state(self) -> dict[str, object]:
        return {
            "templates": [[template.clone() for template in page] for page in self._templates],
            "delta_banks": self._delta_banks,
            "delta_selectors": self._delta_selectors,
            "exceptions": self._exceptions,
            "selector_state": self._selectors.state(),
            "required_pool_size": self.required_pool_size,
        }

    def set_extra_state(self, state: dict[str, object]) -> None:
        self._templates = [
            [template.clone().to(dtype=torch.long, device="cpu") for template in page]
            for page in state["templates"]  # type: ignore[index]
        ]
        self._delta_banks = [
            [tuple((int(offset), int(scalar)) for offset, scalar in delta) for delta in bank]
            for bank in state["delta_banks"]  # type: ignore[index]
        ]
        self._delta_selectors = {
            int(key): int(value)
            for key, value in dict(state["delta_selectors"]).items()  # type: ignore[arg-type]
        }
        self._exceptions = {
            int(key): {int(offset): int(scalar) for offset, scalar in dict(value).items()}
            for key, value in dict(state["exceptions"]).items()  # type: ignore[arg-type]
        }
        selector_state = state.get("selector_state")
        if selector_state is not None:
            self._selectors.load_state(selector_state)  # type: ignore[arg-type]
        self.required_pool_size = int(state["required_pool_size"])
        self._rebuild_intern_lookups()
        self.rebuild_indexes()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):  # type: ignore[no-untyped-def]
        legacy_key = prefix + "base_selectors"
        legacy_selectors = state_dict.pop(legacy_key, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if legacy_selectors is not None:
            self._selectors.load_legacy_dense(legacy_selectors)
            self.rebuild_indexes()
