from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DynamicTransformerConfig:
    """Configuration for the scalar-routed Dynamic Transformer reference model."""

    vocab_size: int = 64
    d_model: int = 16
    n_heads: int = 4
    n_layers: int = 1
    ffn_dim: int = 32
    max_seq_len: int = 64
    dropout: float = 0.0

    # Fraction of scalar route slots that all tokens share at initialization.
    # Example: 0.5 means two 100-slot routes initially share 50 scalars.
    initial_shared_fraction: float = 0.5

    # Scalar pools are preallocated so split operations do not resize Parameters
    # or invalidate Adam state tensors.
    pool_growth_factor: float = 1.5
    init_std: float = 0.02

    def validate(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.d_model <= 0 or self.ffn_dim <= 0:
            raise ValueError("d_model and ffn_dim must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.initial_shared_fraction <= 1.0:
            raise ValueError("initial_shared_fraction must be in [0, 1]")
        if self.pool_growth_factor < 1.0:
            raise ValueError("pool_growth_factor must be >= 1")
