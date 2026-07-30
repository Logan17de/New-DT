from __future__ import annotations

import math
from collections import Counter

import torch
from torch import Tensor


class AdaptiveSelectorTable:
    """Compact token/page template selectors with sparse page divergence.

    The initial DT assignment chooses one template group per token and reuses that
    choice across every route page. Storing ``vocab_size * num_pages`` INT32 values
    would immediately recreate the routing-memory problem, so the common selector
    is stored once per token. Only page-specific deviations are recorded.

    A heavily modified page automatically promotes from a sparse ``token ->
    template`` map to one dense CPU vector. Dense pages can later demote again when
    compaction restores similarity to the token defaults.
    """

    def __init__(
        self,
        vocab_size: int,
        num_pages: int,
        templates_per_page: int,
        *,
        dense_promotion_fraction: float = 0.35,
        dense_demotion_fraction: float = 0.15,
    ) -> None:
        if vocab_size <= 0 or num_pages <= 0 or templates_per_page <= 0:
            raise ValueError("selector dimensions must be positive")
        if not 0 < dense_demotion_fraction < dense_promotion_fraction <= 1:
            raise ValueError("selector dense thresholds are inconsistent")

        self.vocab_size = vocab_size
        self.num_pages = num_pages
        self.templates_per_page = min(templates_per_page, vocab_size)
        self.dense_promotion_fraction = dense_promotion_fraction
        self.dense_demotion_fraction = dense_demotion_fraction

        defaults = torch.empty(vocab_size, dtype=torch.int32)
        for token_id in range(vocab_size):
            selector = min(
                self.templates_per_page - 1,
                token_id * self.templates_per_page // vocab_size,
            )
            defaults[token_id] = selector
        self.token_defaults = defaults
        self._sparse_pages: dict[int, dict[int, int]] = {}
        self._dense_pages: dict[int, Tensor] = {}

    def _validate(self, token_id: int, page_id: int) -> None:
        if not 0 <= token_id < self.vocab_size:
            raise IndexError("token_id outside vocabulary")
        if not 0 <= page_id < self.num_pages:
            raise IndexError("page_id outside selector table")

    def get(self, token_id: int, page_id: int) -> int:
        self._validate(token_id, page_id)
        dense = self._dense_pages.get(page_id)
        if dense is not None:
            return int(dense[token_id].item())
        sparse = self._sparse_pages.get(page_id)
        if sparse is not None and token_id in sparse:
            return int(sparse[token_id])
        return int(self.token_defaults[token_id].item())

    def set(self, token_id: int, page_id: int, template_id: int) -> bool:
        self._validate(token_id, page_id)
        if template_id < 0:
            raise ValueError("template_id must be non-negative")
        old = self.get(token_id, page_id)
        if old == template_id:
            return False

        dense = self._dense_pages.get(page_id)
        if dense is not None:
            dense[token_id] = int(template_id)
            self.compact_page(page_id)
            return True

        default = int(self.token_defaults[token_id].item())
        sparse = self._sparse_pages.setdefault(page_id, {})
        if template_id == default:
            sparse.pop(token_id, None)
        else:
            sparse[token_id] = int(template_id)
        if not sparse:
            self._sparse_pages.pop(page_id, None)
            return True

        if len(sparse) / self.vocab_size >= self.dense_promotion_fraction:
            values = self.token_defaults.clone()
            for changed_token, changed_template in sparse.items():
                values[changed_token] = int(changed_template)
            self._dense_pages[page_id] = values
            self._sparse_pages.pop(page_id, None)
        return True

    def values_for_page(self, page_id: int) -> Tensor:
        if not 0 <= page_id < self.num_pages:
            raise IndexError("page_id outside selector table")
        dense = self._dense_pages.get(page_id)
        if dense is not None:
            return dense
        sparse = self._sparse_pages.get(page_id)
        if not sparse:
            return self.token_defaults
        values = self.token_defaults.clone()
        for token_id, template_id in sparse.items():
            values[token_id] = int(template_id)
        return values

    def page_overrides(self, page_id: int) -> dict[int, int]:
        """Return exact selectors that differ from the per-token defaults."""

        dense = self._dense_pages.get(page_id)
        if dense is not None:
            changed = (dense != self.token_defaults).nonzero(as_tuple=False).flatten()
            return {
                int(token_id): int(dense[token_id].item())
                for token_id in changed.tolist()
            }
        return dict(self._sparse_pages.get(page_id, {}))

    def override_pages(self) -> tuple[int, ...]:
        return tuple(sorted(set(self._sparse_pages) | set(self._dense_pages)))

    def template_counts(self, page_id: int) -> Counter[int]:
        values = self.values_for_page(page_id)
        counts = torch.bincount(values.to(dtype=torch.long))
        return Counter(
            {
                template_id: int(count)
                for template_id, count in enumerate(counts.tolist())
                if count
            }
        )

    def compact_page(self, page_id: int) -> str | None:
        dense = self._dense_pages.get(page_id)
        if dense is None:
            return None
        changed = (dense != self.token_defaults).nonzero(as_tuple=False).flatten()
        if changed.numel() / self.vocab_size > self.dense_demotion_fraction:
            return None
        sparse = {
            int(token_id): int(dense[token_id].item())
            for token_id in changed.tolist()
        }
        self._dense_pages.pop(page_id, None)
        if sparse:
            self._sparse_pages[page_id] = sparse
        return "sparse"

    def max_template_id(self) -> int:
        maximum = int(self.token_defaults.max().item()) if self.vocab_size else 0
        for sparse in self._sparse_pages.values():
            if sparse:
                maximum = max(maximum, max(sparse.values()))
        for dense in self._dense_pages.values():
            if dense.numel():
                maximum = max(maximum, int(dense.max().item()))
        return maximum

    def packed_storage_bits(self) -> dict[str, int]:
        """Estimate exact packed selector payload without Python-object overhead."""

        selector_bits = max(1, math.ceil(math.log2(max(2, self.max_template_id() + 1))))
        token_bits = max(1, math.ceil(math.log2(max(2, self.vocab_size))))
        page_bits = max(1, math.ceil(math.log2(max(2, self.num_pages))))
        default_bits = self.vocab_size * selector_bits
        override_bits = 0
        override_entries = 0
        for page_id in self.override_pages():
            overrides = self.page_overrides(page_id)
            if not overrides:
                continue
            override_entries += len(overrides)
            override_bits += page_bits + len(overrides) * (token_bits + selector_bits)
        return {
            "selector_bits": default_bits + override_bits,
            "selector_default_bits": default_bits,
            "selector_override_bits": override_bits,
            "selector_width_bits": selector_bits,
            "selector_override_entries": override_entries,
        }

    def state(self) -> dict[str, object]:
        return {
            "token_defaults": self.token_defaults.clone(),
            "sparse_pages": {
                int(page): dict(values) for page, values in self._sparse_pages.items()
            },
            "dense_pages": {
                int(page): values.clone() for page, values in self._dense_pages.items()
            },
        }

    def load_state(self, state: dict[str, object]) -> None:
        defaults = state["token_defaults"]
        if not torch.is_tensor(defaults) or defaults.shape != (self.vocab_size,):
            raise ValueError("invalid selector token defaults")
        self.token_defaults = defaults.to(dtype=torch.int32, device="cpu").clone()
        self._sparse_pages = {
            int(page): {int(token): int(template) for token, template in dict(values).items()}
            for page, values in dict(state.get("sparse_pages", {})).items()
        }
        self._dense_pages = {
            int(page): values.to(dtype=torch.int32, device="cpu").clone()
            for page, values in dict(state.get("dense_pages", {})).items()
        }

    def load_legacy_dense(self, selectors: Tensor) -> None:
        """Convert an earlier dense selector checkpoint without retaining it."""

        if selectors.shape != (self.vocab_size, self.num_pages):
            raise ValueError("legacy selector tensor has an unexpected shape")
        dense = selectors.to(dtype=torch.int32, device="cpu")
        self.token_defaults = dense[:, 0].clone()
        self._sparse_pages.clear()
        self._dense_pages.clear()
        for page_id in range(self.num_pages):
            values = dense[:, page_id]
            changed = (values != self.token_defaults).nonzero(as_tuple=False).flatten()
            if not changed.numel():
                continue
            fraction = changed.numel() / self.vocab_size
            if fraction >= self.dense_promotion_fraction:
                self._dense_pages[page_id] = values.clone()
            else:
                self._sparse_pages[page_id] = {
                    int(token_id): int(values[token_id].item())
                    for token_id in changed.tolist()
                }
