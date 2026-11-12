# Guide for coding agents

This repository is developed by agents, and Darkplane AutoMerge merges their pull requests
without human review. Only the goal matters: everything in the repository, including the
benchmark, the eval, the simulator and CI, may be changed when that serves the goal.

## Goal

Compile sparse ternary neural networks into memory-free LUT circuits that are
**correct**, **small** (LUT count) and **shallow** (logic depth), and keep improving them.

Pipeline: `model.py` (ternary network) -> `compiler.py` -> `netlist.py` (what we produce)
-> `simulator.py` -> `evaluate.py`. The compiler's module docstring lists known headroom.

## How to work

1. **A wrong circuit is not progress.** Compiled circuits must stay equivalent to their
   reference networks; a smaller circuit that computes something else does not count.
2. **Change the measurement when it is wrong, not when it is inconvenient.** Change the
   benchmark suite, eval, simulator or regression gate when they measure the goal badly,
   for example a missing realistic case, a too-slow or too-coarse simulation, or a gate that
   blocks a worthwhile LUT/depth trade-off. Say in the pull request why the new measurement
   is closer to the goal. Making a number look better without the circuits getting better
   is not progress.
3. **Show the delta.** Put the benchmark before and after in every pull request description.
4. **Keep pull requests small and single-purpose**, so each one can be reverted on its own.
5. Run `ruff check . && ruff format --check . && pytest -q && python benchmarks/run.py` before
   opening a pull request.

## The README

The owner reads only README.md. Keep it about the goal and the algorithms, in plain language:
no code, file names, function names, commands or setup steps. When a merged change alters how
an algorithm works or moves the results, update the README's algorithm description, the
progress table and the ideas ahead in the same pull request.

## Setup and commands

- Install: `pip install -e ".[dev]"` (add `[telemetry]` to export metrics).
- Check: `ruff check . && ruff format --check . && pytest -q`.
- Benchmark: `python benchmarks/run.py --output results.json`; compare two runs with
  `python benchmarks/compare.py base.json head.json`.
- CI exports results from `main` to Dash0 as `dark_algorithms.benchmark.*` metrics using the
  repository secret `DASH0_AUTH_TOKEN` and the repository variable `DASH0_OTLP_ENDPOINT`
  (the OTLP/HTTP base URL, no path).

## Known limits

- The Dash0 GitHub App cannot push changes under `.github/workflows/` (GitHub refuses it
  without the `workflows` permission). Propose CI changes in the pull request description
  instead; a maintainer has to commit them.
