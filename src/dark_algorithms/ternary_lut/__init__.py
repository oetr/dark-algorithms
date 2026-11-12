"""Ternary neural networks compiled to memory-free LUT circuits.

Pipeline: TernaryNetwork (model) -> compile_network (compiler) -> Netlist (what we
produce) -> simulate (simulator) -> evaluate (correctness and quality metrics).
"""

from .compiler import DEFAULT_K, compile_network
from .evaluate import EvalResult, check_equivalence, evaluate
from .model import TernaryLayer, TernaryNetwork, random_network
from .netlist import Lut, Netlist
from .simulator import simulate

__all__ = [
    "DEFAULT_K",
    "EvalResult",
    "Lut",
    "Netlist",
    "TernaryLayer",
    "TernaryNetwork",
    "check_equivalence",
    "compile_network",
    "evaluate",
    "random_network",
    "simulate",
]
