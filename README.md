# dark-algorithms

An app that continuously improves implementations of state-of-the-art algorithms,
developed by coding agents and cleared by Darkplane AutoMerge.

The long-term goal is to research and pick algorithms with high impact, implement their
state-of-the-art versions, and keep improving them. Version one hard-codes a single
algorithm so the evaluation, abstractions and feedback loop can be built properly first.

## Algorithm 1: ternary neural networks as memory-free LUT circuits

A ternary network has weights in {-1, 0, +1}, bit inputs and threshold activations, so it
is a boolean function. Instead of storing weights in memory, every neuron is compiled
directly into FPGA lookup tables (LUTs). Zero weights vanish from the circuit entirely,
so sparse networks get much smaller.

| Piece | Module | Role |
|---|---|---|
| Model | `ternary_lut/model.py` | Ternary network and its reference semantics |
| Netlist | `ternary_lut/netlist.py` | What we produce: a DAG of k-input LUTs, no memory cells |
| Compiler | `ternary_lut/compiler.py` | Network to netlist, exploiting sparsity |
| Simulator | `ternary_lut/simulator.py` | Fast LUT-level functional simulation, batched with NumPy |
| Eval | `ternary_lut/evaluate.py` | Equivalence check plus LUT count, depth, and savings vs. a dense baseline |

The simulator works at LUT level, not gate or timing level: correctness is checked
functionally (exhaustively for up to 16 inputs) and logic depth stands in for timing.

## Measuring progress

`benchmarks/run.py` compiles a fixed suite of networks and reports, per case, whether the
circuit is equivalent, its LUT count and depth, and the reduction against the dense
baseline. CI runs it on every pull request and fails if the total LUT count or worst-case
depth regresses against the base branch.

On every commit to `main`, CI exports the results to Dash0 as metrics
(`dark_algorithms.benchmark.*`) for the progress dashboard. To turn this on, set:

- repository secret `DASH0_AUTH_TOKEN`: a Dash0 auth token with ingest permission,
- repository variable `DASH0_OTLP_ENDPOINT`: the OTLP/HTTP endpoint from Dash0's
  endpoint settings, e.g. `https://ingress.eu-west-1.aws.dash0.com`.

## How it improves itself

A scheduled Agent0 automation picks one improvement at a time, implements it with tests,
opens a pull request with the before/after benchmark, and gets it merged once CI is green.
Every idea it tries, including failed ones, is logged in [JOURNAL.md](JOURNAL.md). Progress
is charted on the "dark-algorithms progress" dashboard in Dash0.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest -q
python benchmarks/run.py
```

Agents: read [AGENTS.md](AGENTS.md) first.
