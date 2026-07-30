from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn


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

    A logical route is split into independently resolvable pages. Every token/page
    selects an immutable base template. Repeated sparse changes can be interned as
    a shared delta, while one-off changes remain token-specific exceptions. Large
    deltas are absorbed into a new immutable template during compaction.

    Pages are a storage/decoding unit only. Scalar sharing remains exact and may
    occur at arbitrary offsets across unrelated token pages.
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
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or route_size <= 0:
            raise ValueError("vocab_size and route_size must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if templates_per_page <= 0:
            raise ValueError("templates_per_page must be positive")

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

        selectors = torch.empty(vocab_size, self.num_pages, dtype=torch.int32)
        for token_id in range(vocab_size):
            selector = min(
                self.templates_per_page - 1,
                token_id * self.templates_per_page // vocab_size,
            )
            selectors[token_id].fill_(selector)
        self.register_buffer("base_selectors", selectors)

        self._templates: list[list[Tensor]] = []
        self._delta_banks: list[list[tuple[tuple[int, int], ...]]] = [
            [] for _ in range(self.num_pages)
        ]
        self._delta_selectors: dict[int, int] = {}
        self._exceptions: dict[int, dict[int, int]] = {}

        # Program-derived exact reverse index. Base ownership is represented by
        # template users + scalar offsets; only deviations are indexed per route.
        self._template_users: list[dict[int, set[int]]] = []
        self._template_scalar_offsets: list[list[dict[int, tuple[int, ...]]]] = []
        self._explicit_locations: set[int] = set()
        self._page_explicit_locations: dict[int, set[int]] = {}
        self._explicit_to_scalar: dict[int, int] = {}
        self._patch_to_locations: dict[int, set[int]] = defaultdict(set)
        self._usage_counts: Counter[int] = Counter()

        self.required_pool_size = self._initialize_templates()
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
        """Create compact initial route programs without a dense vocab×route table."""

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

    def _template_id(self, token_id: int, page_id: int) -> int:
        return int(self.base_selectors[token_id, page_id].item())

    def _base_page(self, token_id: int, page_id: int) -> Tensor:
        template_id = self._template_id(token_id, page_id)
        return self._templates[page_id][template_id]

    def _delta(self, token_id: int, page_id: int) -> tuple[tuple[int, int], ...]:
        page_instance = self._page_instance(token_id, page_id)
        delta_id = self._delta_selectors.get(page_instance)
        if delta_id is None:
            return ()
        return self._delta_banks[page_id][delta_id]

    def resolve_page(
        self,
        token_id: int,
        page_id: int,
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Decode exactly one page; no earlier or later page is touched."""

        if not 0 <= token_id < self.vocab_size:
            raise IndexError("token_id outside vocabulary")
        if not 0 <= page_id < self.num_pages:
            raise IndexError("page_id outside route")

        route = self._base_page(token_id, page_id).clone()
        for offset, scalar in self._delta(token_id, page_id):
            route[offset] = scalar
        page_instance = self._page_instance(token_id, page_id)
        for offset, scalar in self._exceptions.get(page_instance, {}).items():
            route[offset] = scalar
        if device is not None:
            route = route.to(device=device)
        return route

    def resolve_token(
        self, token_id: int, *, device: torch.device | str | None = None
    ) -> Tensor:
        pages = [
            self.resolve_page(token_id, page_id, device=device)
            for page_id in range(self.num_pages)
        ]
        return torch.cat(pages, dim=0)

    def resolve(self, token_ids: Tensor, *, device: torch.device | str | None = None) -> Tensor:
        flat_ids = token_ids.detach().reshape(-1).cpu().tolist()
        cache: dict[int, Tensor] = {}
        rows: list[Tensor] = []
        for token_id in flat_ids:
            token_id = int(token_id)
            if token_id not in cache:
                cache[token_id] = self.resolve_token(token_id, device=device)
            rows.append(cache[token_id])
        resolved = torch.stack(rows, dim=0)
        return resolved.view(*token_ids.shape, self.route_size)

    def materialize_all(self, *, device: torch.device | str | None = None) -> Tensor:
        return torch.stack(
            [self.resolve_token(token_id, device=device) for token_id in range(self.vocab_size)],
            dim=0,
        )

    def scalar_at(self, token_id: int, route_slot: int) -> int:
        if not 0 <= route_slot < self.route_size:
            raise IndexError("route_slot outside parameter tensor")
        page_id, offset = divmod(route_slot, self.page_size)
        page_instance = self._page_instance(token_id, page_id)
        exception = self._exceptions.get(page_instance, {}).get(offset)
        if exception is not None:
            return exception
        for delta_offset, scalar in self._delta(token_id, page_id):
            if delta_offset == offset:
                return scalar
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

    def _build_template_indexes(self) -> None:
        self._template_users = []
        self._template_scalar_offsets = []
        for page_id, templates in enumerate(self._templates):
            users: dict[int, set[int]] = defaultdict(set)
            for token_id in range(self.vocab_size):
                users[self._template_id(token_id, page_id)].add(token_id)
            self._template_users.append(dict(users))

            indexed_templates: list[dict[int, tuple[int, ...]]] = []
            for template in templates:
                offsets: dict[int, list[int]] = defaultdict(list)
                for offset, scalar in enumerate(template.tolist()):
                    offsets[int(scalar)].append(offset)
                indexed_templates.append(
                    {scalar: tuple(items) for scalar, items in offsets.items()}
                )
            self._template_scalar_offsets.append(indexed_templates)

    def rebuild_indexes(self) -> None:
        """Rebuild compact reverse indexes after initialization/checkpoint loading."""

        self._build_template_indexes()
        self._usage_counts.clear()
        self._explicit_locations.clear()
        self._page_explicit_locations.clear()
        self._explicit_to_scalar.clear()
        self._patch_to_locations.clear()

        for page_id, templates in enumerate(self._templates):
            for template_id, template in enumerate(templates):
                user_count = len(self._template_users[page_id].get(template_id, ()))
                if not user_count:
                    continue
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

    def _add_explicit(
        self, token_id: int, page_id: int, offset: int, scalar: int
    ) -> None:
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
                underlying = scalar
                break

        exceptions = self._exceptions.setdefault(page_instance, {})
        if new_scalar == underlying:
            exceptions.pop(offset, None)
            if not exceptions:
                self._exceptions.pop(page_instance, None)
        else:
            exceptions[offset] = int(new_scalar)

        self._usage_counts[old_scalar] -= 1
        self._usage_counts[new_scalar] += 1
        packed = self._pack_location(token_id, route_slot)
        self._remove_explicit(packed)
        base_scalar = int(self._base_page(token_id, page_id)[offset].item())
        if new_scalar != base_scalar:
            self._add_explicit(token_id, page_id, offset, int(new_scalar))

        if compact:
            self.compact_page(token_id, page_id)
        return True

    def _normalized_exceptions(self, page_instance: int) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self._exceptions.get(page_instance, {}).items()))

    def _intern_delta(
        self, page_id: int, delta: tuple[tuple[int, int], ...]
    ) -> int:
        bank = self._delta_banks[page_id]
        try:
            return bank.index(delta)
        except ValueError:
            bank.append(delta)
            return len(bank) - 1

    def _intern_template(self, page_id: int, route: Tensor) -> int:
        route_cpu = route.detach().to(dtype=torch.long, device="cpu").contiguous()
        for template_id, existing in enumerate(self._templates[page_id]):
            if torch.equal(existing, route_cpu):
                return template_id
        self._templates[page_id].append(route_cpu)
        offsets: dict[int, list[int]] = defaultdict(list)
        for offset, scalar in enumerate(route_cpu.tolist()):
            offsets[int(scalar)].append(offset)
        self._template_scalar_offsets[page_id].append(
            {scalar: tuple(items) for scalar, items in offsets.items()}
        )
        return len(self._templates[page_id]) - 1

    def _set_template_selector(
        self, token_id: int, page_id: int, template_id: int
    ) -> None:
        old_template = self._template_id(token_id, page_id)
        if old_template == template_id:
            return
        self._template_users[page_id].setdefault(old_template, set()).discard(token_id)
        self._template_users[page_id].setdefault(template_id, set()).add(token_id)
        self.base_selectors[token_id, page_id] = template_id

    def compact_page(self, token_id: int, page_id: int, *, force: bool = False) -> str | None:
        """Promote repeated patches to deltas or absorb large patches into templates."""

        page_instance = self._page_instance(token_id, page_id)
        exceptions = self._normalized_exceptions(page_instance)
        if not exceptions:
            return None

        page_length = self._page_length(page_id)
        large_patch = (
            len(exceptions) >= self.template_promotion_threshold
            or len(exceptions) / page_length >= self.template_promotion_fraction
        )
        if force or large_patch:
            resolved = self.resolve_page(token_id, page_id)
            template_id = self._intern_template(page_id, resolved)
            self._set_template_selector(token_id, page_id, template_id)
            self._delta_selectors.pop(page_instance, None)
            self._exceptions.pop(page_instance, None)
            self._refresh_page_explicit_index(token_id, page_id)
            return "template"

        if len(exceptions) < self.delta_promotion_threshold:
            return None

        matching_instances = [
            candidate
            for candidate in self._exceptions
            if candidate % self.num_pages == page_id
            and self._normalized_exceptions(candidate) == exceptions
        ]
        if len(matching_instances) < self.shared_delta_min_reuse:
            return None

        delta_id = self._intern_delta(page_id, exceptions)
        for candidate in matching_instances:
            candidate_token, candidate_page = self._split_page_instance(candidate)
            self._delta_selectors[candidate] = delta_id
            self._exceptions.pop(candidate, None)
            self._refresh_page_explicit_index(candidate_token, candidate_page)
        return "delta"

    def compact_all(self, *, force: bool = False) -> dict[str, int]:
        counts = {"template": 0, "delta": 0}
        for page_instance in tuple(self._exceptions):
            token_id, page_id = self._split_page_instance(page_instance)
            result = self.compact_page(token_id, page_id, force=force)
            if result is not None:
                counts[result] += 1
        return counts

    def route_locations(self, scalar: int) -> tuple[tuple[int, int], ...]:
        """Enumerate exact owners from template indexes plus sparse deviations."""

        locations = set(self._patch_to_locations.get(scalar, ()))
        for page_id, templates in enumerate(self._template_scalar_offsets):
            for template_id, scalar_offsets in enumerate(templates):
                offsets = scalar_offsets.get(scalar)
                if not offsets:
                    continue
                for token_id in self._template_users[page_id].get(template_id, ()):
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
        """Estimate packed payload size; excludes Python-object overhead."""

        neuron_bits = max(1, math.ceil(math.log2(max(pool_size, 2))))
        max_templates = max(len(items) for items in self._templates)
        selector_bits = max(1, math.ceil(math.log2(max(max_templates, 2))))
        offset_bits = max(1, math.ceil(math.log2(max(self.page_size, 2))))

        selector_payload = self.vocab_size * self.num_pages * selector_bits
        template_values = sum(
            template.numel() for page in self._templates for template in page
        )
        template_payload = template_values * neuron_bits
        delta_entries = sum(len(delta) for bank in self._delta_banks for delta in bank)
        delta_payload = delta_entries * (offset_bits + neuron_bits)
        exception_entries = sum(len(items) for items in self._exceptions.values())
        exception_payload = exception_entries * (offset_bits + neuron_bits)
        delta_selector_bits = len(self._delta_selectors) * max(
            1,
            math.ceil(
                math.log2(max(2, max((len(bank) for bank in self._delta_banks), default=1)))
            ),
        )
        total_bits = (
            selector_payload
            + template_payload
            + delta_payload
            + exception_payload
            + delta_selector_bits
        )
        return {
            "selector_bits": selector_payload,
            "template_bits": template_payload,
            "delta_bits": delta_payload + delta_selector_bits,
            "exception_bits": exception_payload,
            "total_bits": total_bits,
            "total_bytes": math.ceil(total_bits / 8),
            "neuron_id_bits": neuron_bits,
            "selector_width_bits": selector_bits,
        }

    def get_extra_state(self) -> dict[str, object]:
        return {
            "templates": [[template.clone() for template in page] for page in self._templates],
            "delta_banks": self._delta_banks,
            "delta_selectors": self._delta_selectors,
            "exceptions": self._exceptions,
            "required_pool_size": self.required_pool_size,
        }

    def set_extra_state(self, state: dict[str, object]) -> None:
        self._templates = [
            [template.clone().to(dtype=torch.long, device="cpu") for template in page]
            for page in state["templates"]  # type: ignore[index]
        ]
        self._delta_banks = state["delta_banks"]  # type: ignore[assignment]
        self._delta_selectors = dict(state["delta_selectors"])  # type: ignore[arg-type]
        self._exceptions = {
            int(key): dict(value)
            for key, value in dict(state["exceptions"]).items()  # type: ignore[arg-type]
        }
        self.required_pool_size = int(state["required_pool_size"])
        self.rebuild_indexes()
