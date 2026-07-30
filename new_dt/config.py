from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DynamicTransformerConfig:
    """Configuration for the scalar-routed Dynamic Transformer."""

    vocab_size: int = 64
    d_model: int = 16
    n_heads: int = 4
    n_layers: int = 1
    ffn_dim: int = 32
    max_seq_len: int = 64
    dropout: float = 0.0

    # Scalar sharing inside each immutable route-template bank.
    initial_shared_fraction: float = 0.5
    pool_growth_factor: float = 1.5
    init_std: float = 0.02

    # Selective Page Reconstruction Compression (SPRC).
    route_page_size: int = 1024
    route_templates_per_page: int = 2
    route_delta_promotion_threshold: int = 32
    route_template_promotion_threshold: int = 256
    route_template_promotion_fraction: float = 0.25
    route_shared_delta_min_reuse: int = 2
    route_cache_pages: int = 256
    route_selector_dense_promotion_fraction: float = 0.35
    route_selector_dense_demotion_fraction: float = 0.15

    # Peak-memory controls. Routed matrices and the LM head are reconstructed in
    # output-row tiles instead of expanding an entire token-owned matrix at once.
    route_linear_out_tile: int = 64
    route_lm_head_tile: int = 1024
    route_materialize_token_chunk: int = 128

    # Rotary position embedding. No additive positional vector is stored.
    rope_theta: float = 10_000.0

    def validate(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.d_model <= 0 or self.ffn_dim <= 0:
            raise ValueError("d_model and ffn_dim must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.initial_shared_fraction <= 1.0:
            raise ValueError("initial_shared_fraction must be in [0, 1]")
        if self.pool_growth_factor < 1.0:
            raise ValueError("pool_growth_factor must be >= 1")
        if self.route_page_size <= 0:
            raise ValueError("route_page_size must be positive")
        if self.route_templates_per_page <= 0:
            raise ValueError("route_templates_per_page must be positive")
        if self.route_delta_promotion_threshold <= 0:
            raise ValueError("route_delta_promotion_threshold must be positive")
        if self.route_template_promotion_threshold <= 0:
            raise ValueError("route_template_promotion_threshold must be positive")
        if not 0 < self.route_template_promotion_fraction <= 1:
            raise ValueError("route_template_promotion_fraction must be in (0, 1]")
        if self.route_shared_delta_min_reuse < 2:
            raise ValueError("route_shared_delta_min_reuse must be at least 2")
        if self.route_cache_pages < 0:
            raise ValueError("route_cache_pages must be non-negative")
        if not (
            0
            < self.route_selector_dense_demotion_fraction
            < self.route_selector_dense_promotion_fraction
            <= 1
        ):
            raise ValueError("selector dense thresholds are inconsistent")
        if self.route_linear_out_tile <= 0 or self.route_lm_head_tile <= 0:
            raise ValueError("route execution tile sizes must be positive")
        if self.route_materialize_token_chunk <= 0:
            raise ValueError("route_materialize_token_chunk must be positive")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
