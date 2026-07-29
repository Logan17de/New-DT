from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import torch

from .model import DynamicTransformer
from .pools import RouteLocation, RoutedParameterTensor


@dataclass(slots=True)
class OwnerGradientStat:
    ema_gradient: float = 0.0
    ema_magnitude: float = 0.0
    samples: int = 0

    def update(self, gradient: float, decay: float) -> None:
        if self.samples == 0:
            self.ema_gradient = gradient
            self.ema_magnitude = abs(gradient)
        else:
            self.ema_gradient = decay * self.ema_gradient + (1 - decay) * gradient
            self.ema_magnitude = decay * self.ema_magnitude + (1 - decay) * abs(gradient)
        self.samples += 1

    def merge_from(self, other: OwnerGradientStat) -> None:
        total = self.samples + other.samples
        if total:
            self.ema_gradient = (
                self.ema_gradient * self.samples + other.ema_gradient * other.samples
            ) / total
            self.ema_magnitude = (
                self.ema_magnitude * self.samples + other.ema_magnitude * other.samples
            ) / total
            self.samples = total


@dataclass(frozen=True, slots=True)
class StructureEvent:
    kind: str
    module: str
    source_scalar: int
    target_scalar: int
    token_id: int | None = None
    route_slot: int | None = None
    owner_count: int | None = None
    score: float | None = None
    threshold: float | None = None


StatKey = tuple[str, int, int, int]


class DynamicStructureController:
    """Delayed structure updates for globally shared scalar neurons.

    Evidence is stored by ``(pool, token, route_slot, scalar)``. Reverse ownership
    and value-bucket indexes make split/merge proportional to affected neurons and
    their actual route locations instead of the complete model.
    """

    def __init__(
        self,
        *,
        structure_interval: int = 100,
        ema_decay: float = 0.95,
        min_owner_samples: int = 8,
        min_gradient_magnitude: float = 1e-5,
        min_conflict_score: float = 0.6,
        owner_threshold_scale: float = 0.03,
        max_conflict_threshold: float = 0.95,
        min_observed_owner_fraction: float = 0.0,
        max_splits_per_pass: int = 8,
        enable_merge: bool = True,
        merge_weight_tolerance: float = 1e-5,
        merge_gradient_tolerance: float = 1e-5,
        merge_min_samples: int = 1,
        max_merge_candidates_per_scalar: int = 32,
        max_merges_per_pass: int = 8,
        optimizer_state_epsilon: float = 1e-8,
    ) -> None:
        if structure_interval <= 0:
            raise ValueError("structure_interval must be positive")
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if not 0 <= min_conflict_score <= max_conflict_threshold <= 1:
            raise ValueError("conflict thresholds must satisfy 0 <= min <= max <= 1")
        if not 0 <= min_observed_owner_fraction <= 1:
            raise ValueError("min_observed_owner_fraction must be in [0, 1]")
        if merge_weight_tolerance <= 0:
            raise ValueError("merge_weight_tolerance must be positive")

        self.structure_interval = structure_interval
        self.ema_decay = ema_decay
        self.min_owner_samples = min_owner_samples
        self.min_gradient_magnitude = min_gradient_magnitude
        self.min_conflict_score = min_conflict_score
        self.owner_threshold_scale = owner_threshold_scale
        self.max_conflict_threshold = max_conflict_threshold
        self.min_observed_owner_fraction = min_observed_owner_fraction
        self.max_splits_per_pass = max_splits_per_pass
        self.enable_merge = enable_merge
        self.merge_weight_tolerance = merge_weight_tolerance
        self.merge_gradient_tolerance = merge_gradient_tolerance
        self.merge_min_samples = merge_min_samples
        self.max_merge_candidates_per_scalar = max_merge_candidates_per_scalar
        self.max_merges_per_pass = max_merges_per_pass
        self.optimizer_state_epsilon = optimizer_state_epsilon

        self.stats: dict[StatKey, OwnerGradientStat] = {}
        self._affected_scalars: dict[str, set[int]] = defaultdict(set)
        self._optimizer_active_scalars: dict[str, set[int]] = defaultdict(set)
        self._merge_buckets: dict[str, dict[int, set[int]]] = {}
        self._scalar_bucket: dict[tuple[str, int], int] = {}
        self._merge_index_initialized: set[str] = set()

    def collect(self, model: DynamicTransformer) -> None:
        """Collect exact route-cell gradients after each backward micro-batch."""
        for module_name, routed in model.routed_tensors():
            for sample in routed.pop_route_gradient_samples():
                grouped: dict[tuple[int, int, int], list[float]] = defaultdict(list)
                rows = zip(
                    sample.token_ids.tolist(), sample.route_slots.tolist(),
                    sample.scalar_ids.tolist(), sample.gradients.tolist(), strict=True,
                )
                for token_id, slot_row, scalar_row, gradient_row in rows:
                    for slot, scalar, gradient in zip(
                        slot_row, scalar_row, gradient_row, strict=True
                    ):
                        scalar = int(scalar)
                        grouped[(int(token_id), int(slot), scalar)].append(float(gradient))
                        self._affected_scalars[module_name].add(scalar)
                        if self.enable_merge:
                            self._optimizer_active_scalars[module_name].add(scalar)
                for (token_id, slot, scalar), gradients in grouped.items():
                    key = (module_name, token_id, slot, scalar)
                    self.stats.setdefault(key, OwnerGradientStat()).update(
                        sum(gradients) / len(gradients), self.ema_decay
                    )

    def maybe_restructure(
        self,
        model: DynamicTransformer,
        optimizer: torch.optim.Optimizer,
        *,
        optimizer_step: int,
        force: bool = False,
    ) -> list[StructureEvent]:
        if not force and optimizer_step % self.structure_interval:
            return []
        modules = dict(model.routed_tensors())
        affected = {
            name: set(ids) for name, ids in self._affected_scalars.items()
            if name in modules and ids
        }
        if not affected:
            return []

        if self.enable_merge:
            self._validate_optimizer(modules, optimizer)
            for name, ids in affected.items():
                routed = modules[name]
                self._ensure_merge_index(name, routed)
                dirty = set(self._optimizer_active_scalars[name]) | ids
                self._refresh_merge_index(name, routed, dirty)
                self._prune_optimizer_active(name, routed, optimizer)

        splits = self._split_conflicts(modules, affected, optimizer)
        protected = {
            (event.module, scalar)
            for event in splits
            for scalar in (event.source_scalar, event.target_scalar)
        }
        events = list(splits)
        if self.enable_merge:
            events.extend(self._merge_redundant(
                modules, affected, optimizer, protected=protected
            ))
        for name in affected:
            self._affected_scalars[name].clear()
        return events

    def _split_threshold(self, owner_count: int) -> float:
        pressure = math.log2(max(1.0, owner_count / 2.0))
        return min(
            self.max_conflict_threshold,
            self.min_conflict_score + self.owner_threshold_scale * pressure,
        )

    def _split_conflicts(
        self,
        modules: dict[str, RoutedParameterTensor],
        affected: dict[str, set[int]],
        optimizer: torch.optim.Optimizer,
    ) -> list[StructureEvent]:
        candidates: list[tuple[float, float, float, str, int, RouteLocation, int, float]] = []
        for name, scalar_ids in affected.items():
            routed = modules[name]
            for scalar in scalar_ids:
                locations = routed.route_locations(scalar)
                owner_count = len(locations)
                if owner_count < 2:
                    continue
                observed = []
                for location in locations:
                    stat = self.stats.get((name, location.token_id, location.route_slot, scalar))
                    if stat is not None and stat.samples >= self.min_owner_samples:
                        observed.append((location, stat))
                if len(observed) < 2:
                    continue
                if len(observed) / owner_count < self.min_observed_owner_fraction:
                    continue
                threshold = self._split_threshold(owner_count)
                total = sum(stat.ema_gradient for _, stat in observed)
                for location, stat in observed:
                    others = (total - stat.ema_gradient) / (len(observed) - 1)
                    if stat.ema_gradient * others >= 0:
                        continue
                    smaller = min(abs(stat.ema_gradient), abs(others))
                    larger = max(abs(stat.ema_gradient), abs(others))
                    if smaller < self.min_gradient_magnitude:
                        continue
                    score = smaller / (larger + 1e-12)
                    if score >= threshold:
                        candidates.append((
                            score - threshold, score, smaller, name, scalar,
                            location, owner_count, threshold,
                        ))

        candidates.sort(reverse=True)
        events: list[StructureEvent] = []
        used_sources: set[tuple[str, int]] = set()
        for _, score, _, name, scalar, location, owner_count, threshold in candidates:
            if len(events) >= self.max_splits_per_pass:
                break
            if (name, scalar) in used_sources:
                continue
            routed = modules[name]
            if routed.scalar_at(location) != scalar:
                continue
            new_scalar = routed.pool.split(scalar, optimizer=optimizer)
            if not routed.reroute_slot(
                location.token_id, location.route_slot, new_scalar,
                expected_old_index=scalar,
            ):
                routed.pool.release(new_scalar, optimizer=optimizer)
                continue
            old_key = (name, location.token_id, location.route_slot, scalar)
            stat = self.stats.pop(old_key, None)
            if stat is not None:
                self.stats[(name, location.token_id, location.route_slot, new_scalar)] = stat
            if self.enable_merge:
                self._optimizer_active_scalars[name].add(new_scalar)
                if name in self._merge_index_initialized:
                    self._add_to_merge_index(name, routed, new_scalar)
            used_sources.add((name, scalar))
            events.append(StructureEvent(
                kind="split", module=name, source_scalar=scalar,
                target_scalar=new_scalar, token_id=location.token_id,
                route_slot=location.route_slot, owner_count=owner_count,
                score=score, threshold=threshold,
            ))
        return events

    def _merge_redundant(
        self,
        modules: dict[str, RoutedParameterTensor],
        affected: dict[str, set[int]],
        optimizer: torch.optim.Optimizer,
        *,
        protected: set[tuple[str, int]],
    ) -> list[StructureEvent]:
        events: list[StructureEvent] = []
        seen: set[tuple[str, int, int]] = set()
        for name, affected_scalars in affected.items():
            if len(events) >= self.max_merges_per_pass:
                break
            routed = modules[name]
            buckets = self._merge_buckets[name]
            for scalar in tuple(affected_scalars):
                if len(events) >= self.max_merges_per_pass:
                    break
                if (name, scalar) in protected or routed.usage_count(scalar) == 0:
                    continue
                bucket = self._scalar_bucket.get((name, scalar))
                if bucket is None:
                    continue
                value = float(routed.pool.values[scalar].detach().cpu())
                nearby: set[int] = set()
                for key in (bucket - 1, bucket, bucket + 1):
                    nearby.update(buckets.get(key, ()))
                nearby.discard(scalar)
                nearby_sorted = sorted(
                    nearby,
                    key=lambda other: abs(
                        float(routed.pool.values[other].detach().cpu()) - value
                    ),
                )[:self.max_merge_candidates_per_scalar]
                for other in nearby_sorted:
                    pair = (name, min(scalar, other), max(scalar, other))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    if (name, other) in protected or routed.usage_count(other) == 0:
                        continue
                    other_value = float(routed.pool.values[other].detach().cpu())
                    if abs(other_value - value) > self.merge_weight_tolerance:
                        continue
                    left_grad, left_samples = self._gradient_summary(name, routed, scalar)
                    right_grad, right_samples = self._gradient_summary(name, routed, other)
                    if min(left_samples, right_samples) < self.merge_min_samples:
                        continue
                    if abs(left_grad - right_grad) > self.merge_gradient_tolerance:
                        continue
                    if routed.usage_count(scalar) <= routed.usage_count(other):
                        source, target = scalar, other
                    else:
                        source, target = other, scalar
                    source_locations = routed.route_locations(source)
                    changed = routed.replace_scalar_everywhere(source, target)
                    if not changed:
                        continue
                    self._redirect_stats(name, source, target, source_locations)
                    self._remove_from_merge_index(name, source)
                    self._refresh_merge_index(name, routed, {target})
                    self._optimizer_active_scalars[name].discard(source)
                    self._optimizer_active_scalars[name].add(target)
                    routed.pool.release(source, optimizer=optimizer)
                    events.append(StructureEvent(
                        kind="merge", module=name, source_scalar=source,
                        target_scalar=target, owner_count=changed,
                        score=abs(left_grad - right_grad),
                        threshold=self.merge_gradient_tolerance,
                    ))
                    break
        return events

    def _gradient_summary(
        self, name: str, routed: RoutedParameterTensor, scalar: int
    ) -> tuple[float, int]:
        weighted, samples = 0.0, 0
        for location in routed.route_locations(scalar):
            stat = self.stats.get((name, location.token_id, location.route_slot, scalar))
            if stat is not None:
                weighted += stat.ema_gradient * stat.samples
                samples += stat.samples
        return (weighted / samples, samples) if samples else (0.0, 0)

    def _redirect_stats(
        self,
        name: str,
        old: int,
        new: int,
        locations: tuple[RouteLocation, ...],
    ) -> None:
        for location in locations:
            old_key = (name, location.token_id, location.route_slot, old)
            stat = self.stats.pop(old_key, None)
            if stat is None:
                continue
            new_key = (name, location.token_id, location.route_slot, new)
            if new_key in self.stats:
                self.stats[new_key].merge_from(stat)
            else:
                self.stats[new_key] = stat

    def _bucket(self, value: float) -> int:
        return math.floor(value / self.merge_weight_tolerance)

    def _ensure_merge_index(self, name: str, routed: RoutedParameterTensor) -> None:
        if name in self._merge_index_initialized:
            return
        self._merge_buckets[name] = defaultdict(set)
        for scalar in routed.iter_used_scalar_indices():
            self._add_to_merge_index(name, routed, scalar)
        self._merge_index_initialized.add(name)

    def _add_to_merge_index(
        self, name: str, routed: RoutedParameterTensor, scalar: int
    ) -> None:
        if routed.usage_count(scalar) == 0:
            return
        bucket = self._bucket(float(routed.pool.values[scalar].detach().cpu()))
        self._merge_buckets[name].setdefault(bucket, set()).add(scalar)
        self._scalar_bucket[(name, scalar)] = bucket

    def _remove_from_merge_index(self, name: str, scalar: int) -> None:
        bucket = self._scalar_bucket.pop((name, scalar), None)
        if bucket is None:
            return
        values = self._merge_buckets[name].get(bucket)
        if values is not None:
            values.discard(scalar)
            if not values:
                self._merge_buckets[name].pop(bucket, None)

    def _refresh_merge_index(
        self, name: str, routed: RoutedParameterTensor, scalars: set[int]
    ) -> None:
        for scalar in scalars:
            self._remove_from_merge_index(name, scalar)
            if bool(routed.pool.active_mask[scalar]) and routed.usage_count(scalar):
                self._add_to_merge_index(name, routed, scalar)

    def _prune_optimizer_active(
        self,
        name: str,
        routed: RoutedParameterTensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        state = optimizer.state.get(routed.pool.values, {})
        avg, avg_sq = state.get("exp_avg"), state.get("exp_avg_sq")
        if not torch.is_tensor(avg) or not torch.is_tensor(avg_sq):
            return
        self._optimizer_active_scalars[name] = {
            scalar for scalar in self._optimizer_active_scalars[name]
            if abs(float(avg[scalar].detach().cpu())) > self.optimizer_state_epsilon
            or abs(float(avg_sq[scalar].detach().cpu())) > self.optimizer_state_epsilon
        }

    def _validate_optimizer(
        self,
        modules: dict[str, RoutedParameterTensor],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        pool_ids = {id(routed.pool.values) for routed in modules.values()}
        for group in optimizer.param_groups:
            if not any(id(parameter) in pool_ids for parameter in group["params"]):
                continue
            if float(group.get("weight_decay", 0.0)):
                raise ValueError(
                    "Incremental merge indexing requires weight_decay=0. "
                    "Use Adam or AdamW(..., weight_decay=0)."
                )
