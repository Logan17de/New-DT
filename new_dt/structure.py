from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from .model import DynamicTransformer
from .pools import RoutedParameterTensor


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
            self.ema_gradient = decay * self.ema_gradient + (1.0 - decay) * gradient
            self.ema_magnitude = decay * self.ema_magnitude + (1.0 - decay) * abs(gradient)
        self.samples += 1


@dataclass(frozen=True, slots=True)
class StructureEvent:
    kind: str
    module: str
    source_scalar: int
    target_scalar: int
    token_id: int | None = None
    score: float | None = None


class DynamicStructureController:
    """Collect owner-specific evidence, then reorganize only between Adam steps.

    Scalar gradients are captured at the gathered route tensor, before autograd
    sums contributions into the shared pool entry. Adam still receives the normal
    summed gradient. The controller only uses the owner-level trace to decide who
    should be rerouted during a delayed split.
    """

    def __init__(
        self,
        *,
        structure_interval: int = 100,
        ema_decay: float = 0.95,
        min_owner_samples: int = 8,
        min_gradient_magnitude: float = 1e-5,
        min_conflict_score: float = 0.6,
        max_splits_per_pass: int = 8,
        enable_merge: bool = True,
        merge_weight_tolerance: float = 1e-5,
        merge_gradient_tolerance: float = 1e-5,
        max_merges_per_pass: int = 8,
    ) -> None:
        if structure_interval <= 0:
            raise ValueError("structure_interval must be positive")
        self.structure_interval = structure_interval
        self.ema_decay = ema_decay
        self.min_owner_samples = min_owner_samples
        self.min_gradient_magnitude = min_gradient_magnitude
        self.min_conflict_score = min_conflict_score
        self.max_splits_per_pass = max_splits_per_pass
        self.enable_merge = enable_merge
        self.merge_weight_tolerance = merge_weight_tolerance
        self.merge_gradient_tolerance = merge_gradient_tolerance
        self.max_merges_per_pass = max_merges_per_pass
        self.stats: dict[tuple[str, int, int], OwnerGradientStat] = {}

    def collect(self, model: DynamicTransformer) -> None:
        """Collect route-owner gradients after each backward micro-batch."""

        for module_name, routed in model.routed_tensors():
            for sample in routed.pop_route_gradient_samples():
                grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
                for token_row, scalar_row, gradient_row in zip(
                    sample.token_ids.tolist(),
                    sample.scalar_ids.tolist(),
                    sample.gradients.tolist(),
                    strict=True,
                ):
                    for scalar_id, gradient in zip(
                        scalar_row, gradient_row, strict=True
                    ):
                        grouped[(int(token_row), int(scalar_id))].append(float(gradient))
                for (token_id, scalar_id), gradients in grouped.items():
                    key = (module_name, token_id, scalar_id)
                    stat = self.stats.setdefault(key, OwnerGradientStat())
                    stat.update(sum(gradients) / len(gradients), self.ema_decay)

    def maybe_restructure(
        self,
        model: DynamicTransformer,
        optimizer: torch.optim.Optimizer,
        *,
        optimizer_step: int,
        force: bool = False,
    ) -> list[StructureEvent]:
        """Run delayed split/merge after Adam; ``force`` is used at training end."""

        if not force and optimizer_step % self.structure_interval != 0:
            return []
        modules = dict(model.routed_tensors())
        events = self._split_conflicts(modules, optimizer)
        if self.enable_merge:
            events.extend(self._merge_redundant(modules, optimizer))
        return events

    def _split_conflicts(
        self,
        modules: dict[str, RoutedParameterTensor],
        optimizer: torch.optim.Optimizer,
    ) -> list[StructureEvent]:
        by_scalar: dict[tuple[str, int], list[tuple[int, OwnerGradientStat]]] = defaultdict(list)
        for (module_name, token_id, scalar_id), stat in self.stats.items():
            if stat.samples >= self.min_owner_samples:
                by_scalar[(module_name, scalar_id)].append((token_id, stat))

        candidates: list[tuple[float, str, int, int]] = []
        for (module_name, scalar_id), owners in by_scalar.items():
            if len(owners) < 2 or module_name not in modules:
                continue
            total = sum(stat.ema_gradient for _, stat in owners)
            for token_id, stat in owners:
                others_mean = (total - stat.ema_gradient) / (len(owners) - 1)
                if stat.ema_gradient * others_mean >= 0:
                    continue
                smaller = min(abs(stat.ema_gradient), abs(others_mean))
                larger = max(abs(stat.ema_gradient), abs(others_mean))
                if smaller < self.min_gradient_magnitude:
                    continue
                score = smaller / (larger + 1e-12)
                if score >= self.min_conflict_score:
                    candidates.append((score, module_name, scalar_id, token_id))

        candidates.sort(reverse=True)
        events: list[StructureEvent] = []
        used_source: set[tuple[str, int]] = set()
        for score, module_name, scalar_id, token_id in candidates:
            if len(events) >= self.max_splits_per_pass:
                break
            if (module_name, scalar_id) in used_source:
                continue
            routed = modules[module_name]
            if token_id not in routed.owners(scalar_id):
                continue
            new_index = routed.pool.split(scalar_id, optimizer=optimizer)
            changed = routed.reroute_token_scalar(token_id, scalar_id, new_index)
            if changed == 0:
                routed.pool.release(new_index, optimizer=optimizer)
                continue

            old_key = (module_name, token_id, scalar_id)
            stat = self.stats.pop(old_key, None)
            if stat is not None:
                self.stats[(module_name, token_id, new_index)] = stat
            used_source.add((module_name, scalar_id))
            events.append(
                StructureEvent(
                    kind="split",
                    module=module_name,
                    source_scalar=scalar_id,
                    target_scalar=new_index,
                    token_id=token_id,
                    score=score,
                )
            )
        return events

    def _merge_redundant(
        self,
        modules: dict[str, RoutedParameterTensor],
        optimizer: torch.optim.Optimizer,
    ) -> list[StructureEvent]:
        events: list[StructureEvent] = []
        for module_name, routed in modules.items():
            if len(events) >= self.max_merges_per_pass:
                break
            used = routed.used_scalar_indices().tolist()
            weighted = sorted(
                (float(routed.pool.values[index].detach().cpu()), int(index))
                for index in used
            )
            for (left_value, left), (right_value, right) in zip(
                weighted, weighted[1:], strict=False
            ):
                if len(events) >= self.max_merges_per_pass:
                    break
                if abs(right_value - left_value) > self.merge_weight_tolerance:
                    continue
                left_owners = routed.owners(left)
                right_owners = routed.owners(right)
                if not left_owners or not right_owners or left_owners & right_owners:
                    continue

                left_grad = self._mean_scalar_gradient(module_name, left)
                right_grad = self._mean_scalar_gradient(module_name, right)
                if abs(left_grad - right_grad) > self.merge_gradient_tolerance:
                    continue

                changed = routed.replace_scalar_everywhere(right, left)
                if changed == 0:
                    continue
                routed.pool.release(right, optimizer=optimizer)
                self._redirect_stats(module_name, right, left)
                events.append(
                    StructureEvent(
                        kind="merge",
                        module=module_name,
                        source_scalar=right,
                        target_scalar=left,
                    )
                )
        return events

    def _mean_scalar_gradient(self, module_name: str, scalar_id: int) -> float:
        values = [
            stat.ema_gradient
            for (name, _, index), stat in self.stats.items()
            if name == module_name and index == scalar_id
        ]
        return sum(values) / len(values) if values else 0.0

    def _redirect_stats(self, module_name: str, old_index: int, new_index: int) -> None:
        replacements: list[tuple[tuple[str, int, int], tuple[str, int, int], OwnerGradientStat]] = []
        for key, stat in list(self.stats.items()):
            name, token_id, scalar_id = key
            if name == module_name and scalar_id == old_index:
                replacements.append((key, (name, token_id, new_index), stat))
        for old_key, new_key, stat in replacements:
            self.stats.pop(old_key, None)
            existing = self.stats.get(new_key)
            if existing is None or stat.samples > existing.samples:
                self.stats[new_key] = stat
