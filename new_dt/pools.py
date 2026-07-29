from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor, nn


@dataclass(frozen=True, order=True, slots=True)
class RouteLocation:
    """One exact route cell owned by a token inside a routed parameter tensor."""

    token_id: int
    route_slot: int


@dataclass(frozen=True, slots=True)
class RouteGradientSample:
    """Per-route gradients captured before sharing sums them into pool scalars."""

    token_ids: Tensor
    route_slots: Tensor
    scalar_ids: Tensor
    gradients: Tensor


class ScalarPool(nn.Module):
    """A preallocated, independently trainable pool of scalar parameters.

    Routes may point to the same entry from any token and any vector/matrix slot.
    Autograd sums those contributions and Adam updates the unique scalar once.
    """

    def __init__(
        self,
        capacity: int,
        initial_active: int,
        *,
        init_std: float = 0.02,
        name: str = "pool",
    ) -> None:
        super().__init__()
        if capacity < initial_active or initial_active <= 0:
            raise ValueError("capacity must be >= initial_active > 0")
        self.name = name
        self.values = nn.Parameter(torch.zeros(capacity))
        self.register_buffer("active_mask", torch.zeros(capacity, dtype=torch.bool))
        self.active_mask[:initial_active] = True
        nn.init.normal_(self.values[:initial_active], mean=0.0, std=init_std)

    @property
    def capacity(self) -> int:
        return int(self.values.numel())

    @property
    def active_count(self) -> int:
        return int(self.active_mask.sum().item())

    def first_free_index(self) -> int:
        free = (~self.active_mask).nonzero(as_tuple=False)
        if free.numel() == 0:
            raise RuntimeError(
                f"Scalar pool '{self.name}' is full. Increase pool_growth_factor."
            )
        return int(free[0].item())

    @torch.no_grad()
    def split(
        self,
        source_index: int,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        copy_optimizer_state: bool = True,
    ) -> int:
        """Clone one scalar into a free slot and optionally clone Adam moments."""

        if not bool(self.active_mask[source_index]):
            raise ValueError(f"Cannot split inactive scalar {source_index}")
        new_index = self.first_free_index()
        self.values[new_index].copy_(self.values[source_index])
        self.active_mask[new_index] = True

        if optimizer is not None:
            state = optimizer.state.get(self.values, {})
            for value in state.values():
                if torch.is_tensor(value) and value.shape == self.values.shape:
                    if copy_optimizer_state:
                        value[new_index].copy_(value[source_index])
                    else:
                        value[new_index].zero_()
        return new_index

    @torch.no_grad()
    def release(
        self,
        index: int,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Return an unreferenced scalar slot to the pool's free list."""

        self.active_mask[index] = False
        self.values[index].zero_()
        if optimizer is not None:
            state = optimizer.state.get(self.values, {})
            for value in state.values():
                if torch.is_tensor(value) and value.shape == self.values.shape:
                    value[index].zero_()


class RoutedParameterTensor(nn.Module):
    """Token-owned tensors assembled from globally reusable pool scalars.

    A scalar may be reused at any route slot, not only the matching coordinate in
    another token vector. Two synchronized indexes are maintained:

    * ``route_ids[token, slot] -> scalar`` (forward routing)
    * ``scalar -> {packed token/slot locations}`` (reverse ownership)

    Split and merge therefore update only known route cells instead of scanning
    the complete route table.
    """

    def __init__(
        self,
        vocab_size: int,
        parameter_shape: tuple[int, ...],
        *,
        shared_fraction: float,
        growth_factor: float,
        init_std: float,
        name: str,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.parameter_shape = parameter_shape
        self.route_size = math.prod(parameter_shape)
        self.name = name

        shared_count = int(round(self.route_size * shared_fraction))
        shared_count = min(max(shared_count, 0), self.route_size)
        private_count = self.route_size - shared_count
        initial_active = shared_count + vocab_size * private_count
        capacity = max(initial_active + 8, int(math.ceil(initial_active * growth_factor)))

        route = torch.empty(vocab_size, self.route_size, dtype=torch.long)
        if shared_count:
            route[:, :shared_count] = torch.arange(shared_count)
        cursor = shared_count
        for token_id in range(vocab_size):
            if private_count:
                route[token_id, shared_count:] = torch.arange(
                    cursor, cursor + private_count
                )
                cursor += private_count

        self.pool = ScalarPool(
            capacity,
            initial_active,
            init_std=init_std,
            name=name,
        )
        self.register_buffer("route_ids", route.view(vocab_size, *parameter_shape))
        self.register_buffer(
            "route_use_count",
            torch.zeros(capacity, dtype=torch.long),
            persistent=False,
        )
        self._scalar_to_packed_slots: dict[int, set[int]] = {}
        self._route_grad_cache: list[tuple[Tensor, Tensor, Tensor]] = []
        self.rebuild_route_index()

    def _pack(self, token_id: int, route_slot: int) -> int:
        return token_id * self.route_size + route_slot

    def _unpack(self, packed: int) -> RouteLocation:
        return RouteLocation(
            token_id=packed // self.route_size,
            route_slot=packed % self.route_size,
        )

    @torch.no_grad()
    def rebuild_route_index(self) -> None:
        """Rebuild reverse ownership once after initialization or checkpoint load."""

        flat = self.route_ids.detach().view(self.vocab_size, self.route_size).cpu()
        reverse: dict[int, set[int]] = {}
        for token_id, scalar_row in enumerate(flat.tolist()):
            for route_slot, scalar_id in enumerate(scalar_row):
                reverse.setdefault(int(scalar_id), set()).add(
                    self._pack(token_id, route_slot)
                )
        self._scalar_to_packed_slots = reverse

        counts = torch.zeros(
            self.pool.capacity,
            dtype=torch.long,
            device=self.route_use_count.device,
        )
        for scalar_id, locations in reverse.items():
            counts[scalar_id] = len(locations)
        self.route_use_count.copy_(counts)

    def _load_from_state_dict(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super()._load_from_state_dict(*args, **kwargs)
        self.rebuild_route_index()

    def forward(self, token_ids: Tensor, *, collect_route_grads: bool = False) -> Tensor:
        scalar_ids = self.route_ids[token_ids]
        materialized = self.pool.values[scalar_ids]
        if collect_route_grads and torch.is_grad_enabled():
            materialized.retain_grad()
            self._route_grad_cache.append(
                (token_ids.detach(), scalar_ids.detach(), materialized)
            )
        return materialized

    def materialize_all(self, *, collect_route_grads: bool = False) -> Tensor:
        token_ids = torch.arange(self.vocab_size, device=self.route_ids.device)
        return self.forward(token_ids, collect_route_grads=collect_route_grads)

    def pop_route_gradient_samples(self) -> Iterator[RouteGradientSample]:
        """Return exact token/slot gradients before shared-pool scatter addition."""

        cache, self._route_grad_cache = self._route_grad_cache, []
        for token_ids, scalar_ids, materialized in cache:
            if materialized.grad is None:
                continue
            rows = int(token_ids.numel())
            route_slots = torch.arange(self.route_size).expand(rows, -1)
            yield RouteGradientSample(
                token_ids=token_ids.reshape(-1).cpu(),
                route_slots=route_slots,
                scalar_ids=scalar_ids.reshape(-1, self.route_size).cpu(),
                gradients=materialized.grad.detach()
                .reshape(-1, self.route_size)
                .cpu(),
            )

    def scalar_at(self, location: RouteLocation) -> int:
        flat = self.route_ids.view(self.vocab_size, self.route_size)
        return int(flat[location.token_id, location.route_slot].item())

    @torch.no_grad()
    def reroute_slot(
        self,
        token_id: int,
        route_slot: int,
        new_index: int,
        *,
        expected_old_index: int | None = None,
    ) -> bool:
        """Move exactly one route cell and update both ownership indexes."""

        if not 0 <= token_id < self.vocab_size:
            raise IndexError("token_id outside vocabulary")
        if not 0 <= route_slot < self.route_size:
            raise IndexError("route_slot outside parameter tensor")
        if not bool(self.pool.active_mask[new_index]):
            raise ValueError(f"Cannot route to inactive scalar {new_index}")

        flat = self.route_ids.view(self.vocab_size, self.route_size)
        old_index = int(flat[token_id, route_slot].item())
        if expected_old_index is not None and old_index != expected_old_index:
            return False
        if old_index == new_index:
            return False

        packed = self._pack(token_id, route_slot)
        old_locations = self._scalar_to_packed_slots.get(old_index)
        if old_locations is None or packed not in old_locations:
            raise RuntimeError("reverse route index is inconsistent")

        flat[token_id, route_slot] = new_index
        old_locations.remove(packed)
        if not old_locations:
            self._scalar_to_packed_slots.pop(old_index, None)
        self._scalar_to_packed_slots.setdefault(new_index, set()).add(packed)
        self.route_use_count[old_index] -= 1
        self.route_use_count[new_index] += 1
        return True

    @torch.no_grad()
    def replace_scalar_everywhere(self, old_index: int, new_index: int) -> int:
        """Redirect only locations in the reverse map; never scan all routes."""

        packed_locations = tuple(self._scalar_to_packed_slots.get(old_index, ()))
        changed = 0
        for packed in packed_locations:
            location = self._unpack(packed)
            if self.reroute_slot(
                location.token_id,
                location.route_slot,
                new_index,
                expected_old_index=old_index,
            ):
                changed += 1
        return changed

    def route_locations(self, scalar_index: int) -> tuple[RouteLocation, ...]:
        return tuple(
            sorted(
                self._unpack(packed)
                for packed in self._scalar_to_packed_slots.get(scalar_index, ())
            )
        )

    def owner_tokens(self, scalar_index: int) -> set[int]:
        return {item.token_id for item in self.route_locations(scalar_index)}

    def usage_count(self, scalar_index: int) -> int:
        return int(self.route_use_count[scalar_index].item())

    def iter_used_scalar_indices(self) -> Iterator[int]:
        yield from self._scalar_to_packed_slots.keys()

    def used_scalar_indices(self) -> Tensor:
        return torch.tensor(
            sorted(self._scalar_to_packed_slots),
            dtype=torch.long,
            device=self.route_ids.device,
        )
