from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor, nn

from .routing import RoutePageRecipe, SelectivePageReconstructionStore


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
    """A preallocated, independently trainable pool of scalar parameters."""

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
    """Token-owned tensor reconstructed from immutable route pages.

    Persistent routes use Selective Page Reconstruction Compression (SPRC):

    ``base template + optional shared delta + rare token exception``.

    A split writes one exception. Repeated patches can become shared deltas, and a
    large patch is absorbed into a new immutable template. Scalar sharing remains
    slot-level and exact; pages are only storage and selective-decoding units.
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
        page_size: int = 1024,
        templates_per_page: int = 2,
        delta_promotion_threshold: int = 32,
        template_promotion_threshold: int = 256,
        template_promotion_fraction: float = 0.25,
        shared_delta_min_reuse: int = 2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.parameter_shape = parameter_shape
        self.route_size = math.prod(parameter_shape)
        self.name = name

        self.route_program = SelectivePageReconstructionStore(
            vocab_size,
            self.route_size,
            page_size=page_size,
            templates_per_page=templates_per_page,
            shared_fraction=shared_fraction,
            delta_promotion_threshold=delta_promotion_threshold,
            template_promotion_threshold=template_promotion_threshold,
            template_promotion_fraction=template_promotion_fraction,
            shared_delta_min_reuse=shared_delta_min_reuse,
        )
        initial_active = self.route_program.required_pool_size
        capacity = max(initial_active + 8, int(math.ceil(initial_active * growth_factor)))
        self.pool = ScalarPool(
            capacity,
            initial_active,
            init_std=init_std,
            name=name,
        )
        self._route_grad_cache: list[tuple[Tensor, Tensor, Tensor]] = []

    @property
    def route_ids(self) -> Tensor:
        """Compatibility/debug view. It materializes all routes on demand."""

        return self.route_program.materialize_all(device=self.pool.values.device).view(
            self.vocab_size, *self.parameter_shape
        )

    def rebuild_route_index(self) -> None:
        self.route_program.rebuild_indexes()

    def forward(self, token_ids: Tensor, *, collect_route_grads: bool = False) -> Tensor:
        scalar_ids = self.route_program.resolve(
            token_ids, device=self.pool.values.device
        ).view(*token_ids.shape, *self.parameter_shape)
        materialized = self.pool.values[scalar_ids]
        if collect_route_grads and torch.is_grad_enabled():
            materialized.retain_grad()
            self._route_grad_cache.append(
                (token_ids.detach(), scalar_ids.detach(), materialized)
            )
        return materialized

    def resolve_page(self, token_id: int, page_id: int) -> Tensor:
        """Resolve one logical route page without decoding the complete tensor."""

        return self.route_program.resolve_page(
            token_id, page_id, device=self.pool.values.device
        )

    def page_recipe(self, token_id: int, page_id: int) -> RoutePageRecipe:
        return self.route_program.page_recipe(token_id, page_id)

    def materialize_all(self, *, collect_route_grads: bool = False) -> Tensor:
        token_ids = torch.arange(self.vocab_size, device=self.pool.values.device)
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
        return self.route_program.scalar_at(location.token_id, location.route_slot)

    @torch.no_grad()
    def reroute_slot(
        self,
        token_id: int,
        route_slot: int,
        new_index: int,
        *,
        expected_old_index: int | None = None,
    ) -> bool:
        """Move one exact route cell by creating/removing an SPRC exception."""

        if not 0 <= token_id < self.vocab_size:
            raise IndexError("token_id outside vocabulary")
        if not 0 <= route_slot < self.route_size:
            raise IndexError("route_slot outside parameter tensor")
        if not bool(self.pool.active_mask[new_index]):
            raise ValueError(f"Cannot route to inactive scalar {new_index}")
        return self.route_program.reroute_slot(
            token_id,
            route_slot,
            new_index,
            expected_old_scalar=expected_old_index,
        )

    @torch.no_grad()
    def replace_scalar_everywhere(self, old_index: int, new_index: int) -> int:
        """Redirect exact owners found from templates plus sparse deviations."""

        return self.route_program.replace_scalar_everywhere(old_index, new_index)

    def route_locations(self, scalar_index: int) -> tuple[RouteLocation, ...]:
        return tuple(
            RouteLocation(token_id, route_slot)
            for token_id, route_slot in self.route_program.route_locations(scalar_index)
        )

    def owner_tokens(self, scalar_index: int) -> set[int]:
        return {item.token_id for item in self.route_locations(scalar_index)}

    def usage_count(self, scalar_index: int) -> int:
        return self.route_program.usage_count(scalar_index)

    def iter_used_scalar_indices(self) -> Iterator[int]:
        yield from self.route_program.iter_used_scalars()

    def used_scalar_indices(self) -> Tensor:
        return torch.tensor(
            sorted(self.route_program.iter_used_scalars()),
            dtype=torch.long,
            device=self.pool.values.device,
        )

    def compact_routes(self, *, force: bool = False) -> dict[str, int]:
        return self.route_program.compact_all(force=force)

    def routing_storage_estimate(self) -> dict[str, int]:
        return self.route_program.storage_estimate(pool_size=self.pool.capacity)
