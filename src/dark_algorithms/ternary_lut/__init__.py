"""Ternary neural networks compiled to memory-free LUT circuits.

Pipeline: TernaryNetwork (model) -> compile_network (compiler) -> Netlist (what we
produce) -> simulate (simulator) and prove_equivalence (prove) -> evaluate (correctness and
quality metrics).
"""

from .compiler import DEFAULT_K, compile_network
from .evaluate import EvalResult, check_equivalence, evaluate
from .model import TernaryLayer, TernaryNetwork, random_network
from .netlist import Lut, Netlist
from .prove import ProofResult, prove_equivalence
from .simulator import simulate

__all__ = [
    "DEFAULT_K",
    "EvalResult",
    "Lut",
    "Netlist",
    "ProofResult",
    "TernaryLayer",
    "TernaryNetwork",
    "check_equivalence",
    "compile_network",
    "evaluate",
    "prove_equivalence",
    "random_network",
    "simulate",
]
