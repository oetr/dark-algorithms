# Guide for coding agents

This repository is developed mostly by agents, and pull requests are cleared by Darkplane
AutoMerge. These rules keep that safe.

## Goal

Compile sparse ternary neural networks into memory-free LUT circuits that are
**correct**, **small** (LUT count) and **shallow** (logic depth), and keep improving them.

Pipeline: `model.py` (ternary network) -> `compiler.py` -> `netlist.py` (what we produce)
-> `simulator.py` -> `evaluate.py`. The compiler's module docstring lists known headroom.

## Rules

1. **Correctness is not negotiable.** Every compiled circuit must be equivalent to its
   reference network. Never weaken an equivalence check, a test or the regression gate to
   make a change pass.
2. **Do not touch the yardstick.** `benchmarks/`, `src/dark_algorithms/ternary_lut/evaluate.py`,
   `src/dark_algorithms/ternary_lut/simulator.py` and `.github/` define how progress is measured.
   Changes there are held for human review. Improve the compiler, not the measurement.
3. **Only move the frontier forward.** CI fails a pull request whose total LUT count or
   worst-case depth regresses against the base branch.
4. **Keep pull requests small and single-purpose.** One optimisation per pull request, with
   a test that exercises it and the benchmark delta in the description.
5. Run `ruff check . && ruff format --check . && pytest -q && python benchmarks/run.py` before
   opening a pull request.
