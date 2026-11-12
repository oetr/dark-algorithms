"""Evaluation: is the circuit correct, and how good is it?

Correctness: the simulated netlist must agree with the reference network on
every input vector. Small networks (<= EXHAUSTIVE_MAX_INPUTS inputs) are checked
exhaustively, larger ones on random samples. Every circuit is also proven equivalent
for all inputs with BDDs (see prove.py), time-boxed; `proven` reports whether it was.
A proof that finds a real counterexample makes the circuit not equivalent.

Quality, for a correct circuit:
  * lut_count       - area (lower is better),
  * depth           - logic levels on the critical path, a clock-speed proxy (lower is better),
  * lut_reduction   - LUTs saved against the dense baseline that ignores sparsity.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from .compiler import DEFAULT_K, compile_network
from .model import TernaryNetwork
from .netlist import Netlist
from .prove import prove_equivalence
from .simulator import simulate

EXHAUSTIVE_MAX_INPUTS = 16


@dataclass(frozen=True)
class EquivalenceResult:
    equivalent: bool
    exhaustive: bool
    vectors_checked: int
    mismatching_vectors: int


@dataclass(frozen=True)
class EvalResult:
    equivalent: bool
    exhaustive: bool
    vectors_checked: int
    proven: bool
    proof_seconds: float
    lut_count: int
    depth: int
    baseline_lut_count: int
    baseline_depth: int
    lut_reduction: float
    sparsity: float
    compile_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def check_equivalence(
    network: TernaryNetwork,
    netlist: Netlist,
    num_samples: int = 4096,
    seed: int = 0,
) -> EquivalenceResult:
    n = network.num_inputs
    if n <= EXHAUSTIVE_MAX_INPUTS:
        idx = np.arange(1 << n, dtype=np.int64)
        vectors = ((idx[:, None] >> np.arange(n)) & 1).astype(np.uint8)
        exhaustive = True
    else:
        rng = np.random.default_rng(seed)
        vectors = rng.integers(0, 2, size=(num_samples, n), dtype=np.uint8)
        exhaustive = False
    expected = network.forward(vectors)
    actual = simulate(netlist, vectors)
    mismatches = int(np.any(expected != actual, axis=1).sum())
    return EquivalenceResult(
        equivalent=mismatches == 0,
        exhaustive=exhaustive,
        vectors_checked=int(vectors.shape[0]),
        mismatching_vectors=mismatches,
    )


def evaluate(
    network: TernaryNetwork,
    k: int = DEFAULT_K,
    num_samples: int = 4096,
    seed: int = 0,
    proof_time_box: float = 300.0,
) -> EvalResult:
    """Compile with and without sparsity exploitation, verify, and report quality metrics."""
    start = time.perf_counter()
    netlist = compile_network(network, k=k, prune_zeros=True)
    compile_seconds = time.perf_counter() - start
    baseline = compile_network(network, k=k, prune_zeros=False)
    eq = check_equivalence(network, netlist, num_samples=num_samples, seed=seed)
    proof = prove_equivalence(network, netlist, time_box=proof_time_box)
    base_count = baseline.lut_count()
    return EvalResult(
        equivalent=eq.equivalent and not proof.refuted,
        exhaustive=eq.exhaustive,
        vectors_checked=eq.vectors_checked,
        proven=proof.proven,
        proof_seconds=proof.seconds,
        lut_count=netlist.lut_count(),
        depth=netlist.depth(),
        baseline_lut_count=base_count,
        baseline_depth=baseline.depth(),
        lut_reduction=1.0 - netlist.lut_count() / base_count if base_count else 0.0,
        sparsity=network.sparsity(),
        compile_seconds=compile_seconds,
    )
