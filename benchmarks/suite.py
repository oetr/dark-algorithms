"""The fixed benchmark suite. Results are only comparable across commits if this stays fixed.

Add new cases rather than changing existing ones; the regression gate compares cases by name.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    name: str
    layer_sizes: tuple[int, ...]
    sparsity: float
    seed: int
    k: int = 6


SUITE: tuple[Case, ...] = (
    Case("tiny-exhaustive-s50", (16, 16, 8), 0.5, seed=1),
    Case("mlp-64-s70", (64, 64, 32, 10), 0.7, seed=1),
    Case("mlp-256-s90", (256, 128, 64, 10), 0.9, seed=1),
    Case("mlp-256-s95", (256, 128, 64, 10), 0.95, seed=2),
    Case("mlp-256-s70", (256, 128, 64, 10), 0.7, seed=3),
    Case("mlp-256-s90-k4", (256, 128, 64, 10), 0.9, seed=1, k=4),
)
