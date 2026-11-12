# Guide for coding agents

This repository is developed by agents, and Darkplane AutoMerge merges their pull requests
without human review. Only the goal matters: everything in the repository, including the
benchmark, the eval, the simulator and CI, may be changed when that serves the goal.

## Goal

Compile sparse ternary neural networks into memory-free LUT circuits that are
**correct**, **small** (LUT count) and **shallow** (logic depth), and keep improving them.

Progress is measured against an outside reference, not only against our own earlier
versions: the open-source Yosys/ABC flow synthesizing the same networks to the same
LUT-only target. The headline number is total LUTs relative to Yosys/ABC (below 100% means
we beat it). The stored Yosys results are in `benchmarks/yosys_baseline.json`.

A second, real-FPGA reference uses Yosys's vendor flows with carry chains (`synth_xilinx` for
k=6, `synth_ecp5` for k=4), stored in `benchmarks/fpga_baseline.json` and reported as LUT
sites and carry cells per case. Our k-LUTs map one to one onto those chips' LUT primitives
with no carry cells, so our LUT sites there are our LUT count.

### Current priorities

1. Keep shrinking LUTs and depth, reporting the gap to Yosys/ABC and to the vendor flows in
   every pull request.
2. The vendor-flow result for mlp-256-s70 is missing: `synth_xilinx` takes over nine minutes
   on it, longer than an agent's sandbox allows for one command. Generate it where it can
   run (`python benchmarks/fpga_baseline.py --cases mlp-256-s70 --network-only`).
3. Depth is not yet compared on the vendor targets (a carry chain is much faster than a LUT
   level); a timing-aware comparison, and using carry chains in our own circuits where they
   save LUTs, are open.
4. Six cases are random networks; digits-64-s70 is trained on a real task (same shape and
   sparsity as mlp-64-s70) and has uneven fan-in, off-centre thresholds, dead neurons and much
   more input overlap between neurons. Report it separately when a change helps the random
   cases but not the trained one. Larger trained cases (published FPGA network shapes) are open.

Every benchmark circuit is formally proven equivalent to its network (`prove.py`, reported
as `proven` per case); the regression gate fails when a case the base proved is no longer
proven. A compiler change must keep each neuron's output a separate signal per layer
(`Netlist.layer_outputs`) and its counters recognisable as LUT groups over the same inputs,
or extend the prover so the proof still goes through.

Pipeline: `model.py` (ternary network) -> `compiler.py` -> `netlist.py` (what we produce)
-> `simulator.py` and `prove.py` -> `evaluate.py`. The compiler's module docstring lists known headroom.

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

- Install: `pip install -e ".[dev,yosys]"` (add `telemetry` to export metrics). Yosys comes
  from PyPI as `yowasp-yosys`, a WebAssembly build: no system packages needed, but it only
  sees files under its working directory.
- Yosys baseline: `python benchmarks/yosys_baseline.py [--cases NAME ...]`. Regenerate and
  commit `benchmarks/yosys_baseline.json` whenever the benchmark suite changes. Large cases
  take many minutes (the 70%-sparse 256-input case over 9 minutes, the trained digits case
  12 minutes), so never run it per PR.
- Real-FPGA baseline: `python benchmarks/fpga_baseline.py [--cases NAME ...] [--network-only]`.
  Regenerate `benchmarks/fpga_baseline.json` when the suite changes; its `ours_through_flow`
  entries (our circuit handed to the vendor flow) are informational and go stale as the
  compiler changes. The 256-input cases take 15-35 s each, mlp-256-s70 over 9 minutes and
  digits-64-s70 14 minutes (pass `--timeout 1200`; trained structure slows Yosys down).
- Trained cases: `benchmarks/networks/*.json`, produced by `python benchmarks/train_digits.py`
  (needs the `train` extra, scikit-learn; seeded and reproducible). The benchmark only reads
  the stored file. A new trained case needs its Yosys and vendor-flow baselines generated.
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
