"""Independent baseline: what the open-source Yosys/ABC flow makes of each benchmark network.

    python benchmarks/yosys_baseline.py [--cases NAME ...] [--timeout SECONDS]

Each network is written as plain behavioral Verilog (per neuron: count of +1 inputs >=
count of -1 inputs + threshold) and synthesized with `synth -flatten -lut K`, which maps
to K-input LUTs only, the same rules our compiler follows. The LUT count and logic depth
are stored in benchmarks/yosys_baseline.json, which the benchmark reads.

The networks are fixed by seed, so the baseline only has to be regenerated when the suite
changes or Yosys is upgraded. Large cases can take many minutes.

Yosys comes from PyPI (`pip install yowasp-yosys`, the `yosys` extra), a WebAssembly
build that needs no system packages.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from suite import SUITE  # noqa: E402

from dark_algorithms.ternary_lut import TernaryNetwork, random_network  # noqa: E402

BASELINE_PATH = Path(__file__).resolve().parent / "yosys_baseline.json"


def to_verilog(network: TernaryNetwork) -> str:
    """Behavioral Verilog for the network, with no structure imposed beyond the arithmetic."""
    lines = [
        f"module top(input [{network.num_inputs - 1}:0] x, output [{network.num_outputs - 1}:0] y);"
    ]
    prev = "x"
    for li, layer in enumerate(network.layers):
        cur = "y" if li == len(network.layers) - 1 else f"h{li}"
        if cur != "y":
            lines.append(f"  wire [{layer.out_features - 1}:0] {cur};")
        for j in range(layer.out_features):
            w = layer.weights[j]
            t = int(layer.thresholds[j])
            pos = " + ".join(f"{prev}[{i}]" for i in np.nonzero(w > 0)[0]) or "0"
            neg = " + ".join(f"{prev}[{i}]" for i in np.nonzero(w < 0)[0]) or "0"
            width = max(2, int(np.ceil(np.log2(len(w) + abs(t) + 2))) + 1)
            lines.append(f"  wire [{width - 1}:0] p{li}_{j} = {pos};")
            lines.append(f"  wire [{width - 1}:0] n{li}_{j} = {neg};")
            if t >= 0:
                lines.append(f"  assign {cur}[{j}] = p{li}_{j} >= n{li}_{j} + {width}'d{t};")
            else:
                lines.append(f"  assign {cur}[{j}] = p{li}_{j} + {width}'d{-t} >= n{li}_{j};")
        prev = cur
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def yosys_binary() -> str:
    for name in ("yowasp-yosys", "yosys"):
        path = shutil.which(name)
        if path:
            return path
    sys.exit("Yosys not found: pip install yowasp-yosys (or the `yosys` extra)")


def synthesize(network: TernaryNetwork, k: int, timeout: float) -> dict:
    """Run Yosys/ABC LUT mapping and return LUT count, depth, version and runtime."""
    yosys = yosys_binary()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "net.v"
        src.write_text(to_verilog(network))
        stat, ltp = Path(tmp) / "stat.txt", Path(tmp) / "ltp.txt"
        # Relative paths and cwd: the WebAssembly build only sees its working directory.
        script = (
            f"read_verilog {src.name}; synth -flatten -top top -lut {k}; opt_clean; "
            f"tee -q -o {stat.name} stat; tee -q -o {ltp.name} ltp -noff"
        )
        start = time.perf_counter()
        subprocess.run(
            [yosys, "-q", "-p", script],
            check=True,
            timeout=timeout,
            cwd=tmp,
            stdout=subprocess.DEVNULL,
        )
        seconds = time.perf_counter() - start
        luts = int(re.search(r"(\d+)\s+\$lut", stat.read_text()).group(1))
        depth = int(re.search(r"length=(\d+)", ltp.read_text()).group(1))
        version = subprocess.run([yosys, "-V"], capture_output=True, text=True).stdout.strip()
    return {
        "lut_count": luts,
        "depth": depth,
        "seconds": round(seconds, 1),
        "tool": version.split(" (")[0],
        "flow": f"synth -flatten -lut {k}",
    }


def synthesize_netlist(network: TernaryNetwork, k: int, timeout: float = 600) -> dict:
    """Synthesize like `synthesize` and return Yosys's JSON netlist of the mapped design."""
    return synthesize_verilog_netlist(to_verilog(network), k, timeout)


def synthesize_verilog_netlist(verilog: str, k: int, timeout: float = 600) -> dict:
    """Map a `top` module to k-input LUTs and return Yosys's JSON netlist of it."""
    yosys = yosys_binary()
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "net.v").write_text(verilog)
        script = (
            "read_verilog net.v; synth -flatten -top top -lut %d; opt_clean; write_json out.json"
        )
        subprocess.run(
            [yosys, "-q", "-p", script % k],
            check=True,
            timeout=timeout,
            cwd=tmp,
            stdout=subprocess.DEVNULL,
        )
        return json.loads((Path(tmp) / "out.json").read_text())


def simulate_yosys_netlist(netlist: dict, inputs: np.ndarray) -> np.ndarray:
    """Evaluate a Yosys JSON netlist made of $lut cells on a batch of input bit vectors."""
    mod = netlist["modules"]["top"]
    values: dict = {"0": np.zeros(len(inputs), np.uint8), "1": np.ones(len(inputs), np.uint8)}
    for i, bit in enumerate(mod["ports"]["x"]["bits"]):
        values[bit] = inputs[:, i].astype(np.uint8)
    pending = [c for c in mod["cells"].values() if c["type"] == "$lut"]
    if any(c["type"] != "$lut" for c in mod["cells"].values()):
        raise ValueError("netlist contains cells other than $lut")
    while pending:
        rest = []
        for cell in pending:
            ins = cell["connections"]["A"]
            if all(b in values for b in ins):
                table = cell["parameters"]["LUT"]  # binary string, MSB first
                bits = np.array([int(ch) for ch in reversed(table)], np.uint8)
                idx = np.zeros(len(inputs), np.int64)
                for j, b in enumerate(ins):
                    idx |= values[b].astype(np.int64) << j
                values[cell["connections"]["Y"][0]] = bits[idx]
            else:
                rest.append(cell)
        if len(rest) == len(pending):
            raise ValueError("combinational loop or undriven signal in netlist")
        pending = rest
    out_bits = mod["ports"]["y"]["bits"]
    # Output bits can also be driven directly by inputs or constants via netnames.
    return np.stack([values[b] for b in out_bits], axis=1)


def load_baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", help="only these cases (default: all)")
    parser.add_argument("--timeout", type=float, default=3600, help="seconds per case")
    parser.add_argument("--output", type=Path, default=BASELINE_PATH, help="JSON file to update")
    args = parser.parse_args()
    baseline = json.loads(args.output.read_text()) if args.output.exists() else {}
    for case in SUITE:
        if args.cases and case.name not in args.cases:
            continue
        net = random_network(case.layer_sizes, case.sparsity, np.random.default_rng(case.seed))
        try:
            baseline[case.name] = synthesize(net, case.k, args.timeout)
        except subprocess.TimeoutExpired:
            print(f"{case.name}: Yosys did not finish within {args.timeout:.0f} s", file=sys.stderr)
            continue
        print(f"{case.name}: {baseline[case.name]}")
        args.output.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
