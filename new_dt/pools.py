from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class RouteGradientSample:
    """Per-owner gradients captured before they are summed into a shared scalar."""

    token_ids: Tensor
    scalar_ids: Tensor
    gradients: Tensor


class ScalarPool(nn.Module):
    """A preallocated, independently trainable pool of scalar parameters.

    Routes point to entries in ``values``. Multiple tokens may point to the same
    entry, so autograd naturally sums their gradients and Adam updates that
    unique scalar exactly once.
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
    """Token-owned parameter tensors assembled from shared scalar pool entries.

    For a token-specific matrix with shape ``[out_features, in_features]``, every
    matrix element is an independent route slot pointing to one scalar. This is
    intentionally a clear reference implementation rather than a fast kernel.
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
        self._route_grad_cache: list[tuple[Tensor, Tensor, Tensor]] = []

    def forward(self, token_ids: Tensor, *, collect_route_grads: bool = False) -> Tensor:
        scalar_ids = self.route_ids[token_ids]
        materialized = self.pool.values[scalar_ids]
        if collect_route_grads and torch.is_grad_enabled():
            materialized.retain_grad()
            self._route_grad_cache.append((token_ids.detach(), scalar_ids.detach(), materialized))
        return materialized

    def materialize_all(self, *, collect_route_grads: bool = False) -> Tensor:
        token_ids = torch.arange(self.vocab_size, device=self.route_ids.device)
        return self.forward(token_ids, collect_route_grads=collect_route_grads)

    def pop_route_gradient_samples(self) -> Iterator[RouteGradientSample]:
        """Return gradients per token owner before shared-pool scatter addition."""

        cache, self._route_grad_cache = self._route_grad_cache, []
        for token_ids, scalar_ids, materialized in cache:
            if materialized.grad is None:
                continue
            route_size = self.route_size
            yield RouteGradientSample(
                token_ids=token_ids.reshape(-1).cpu(),
                scalar_ids=scalar_ids.reshape(-1, route_size).cpu(),
                gradients=materialized.grad.detach().reshape(-1, route_size).cpu(),
            )

    @torch.no_grad()
    def reroute_token_scalar(self, token_id: int, old_index: int, new_index: int) -> int:
        token_route = self.route_ids[token_id]
        mask = token_route == old_index
        count = int(mask.sum().item())
        token_route[mask] = new_index
        return count

    @torch.no_grad()
    def replace_scalar_everywhere(self, old_index: int, new_index: int) -> int:
        mask = self.route_ids == old_index
        count = int(mask.sum().item())
        self.route_ids[mask] = new_index
        return count

    def owners(self, scalar_index: int) -> set[int]:
        flat = self.route_ids.view(self.vocab_size, -1)
        owner_ids = (flat == scalar_index).any(dim=1).nonzero(as_tuple=False).flatten()
        return set(int(item) for item in owner_ids.tolist())

    def used_scalar_indices(self) -> Tensor:
        return torch.unique(self.route_ids)
