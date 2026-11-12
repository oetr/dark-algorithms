"""Real-FPGA reference: what Yosys's vendor flows make of each benchmark network.

    python benchmarks/fpga_baseline.py [--cases NAME ...] [--timeout SECONDS] [--network-only]

The LUT-only reference (yosys_baseline.py) maps the networks to K-input LUTs and nothing
else. Real FPGAs also have dedicated carry chains, which make adders and comparators much
cheaper, so a LUT-only count can flatter a compiler that, like ours, avoids adders. This
script synthesizes the same behavioral Verilog with a vendor flow that uses the carry
chains, matched to the case's LUT size:

  * k=6: `synth_xilinx` (7-series: LUT1-LUT6 and CARRY4),
  * k=4: `synth_ecp5` (Lattice ECP5: LUT4 and CCU2C, a carry cell holding two LUT4s).

For each it records the LUT sites the design occupies (Xilinx: LUT1-LUT6 and INV cells;
ECP5: LUT4 cells plus two per CCU2C), the carry cells, and the wide-function muxes, which
use dedicated slice resources and no LUTs.

It also hands our compiled circuit, written as plain LUT logic, to the same flow, to see
whether the vendor's own optimisation can still shrink it. Our k-LUT netlist maps one to
one onto the target's LUT primitives (LUT6 on Xilinx, LUT4 on ECP5) with no carry cells,
so our cost on the target is never more than our own LUT count; the benchmark uses that.

Results go to benchmarks/fpga_baseline.json. Regenerate it whenever the benchmark suite
changes or Yosys is upgraded; the network side does not depend on our compiler.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from suite import SUITE  # noqa: E402
from yosys_baseline import to_verilog, yosys_binary  # noqa: E402

from dark_algorithms.ternary_lut import Netlist, compile_network, random_network  # noqa: E402
from dark_algorithms.ternary_lut.netlist import CONST0, CONST1, FIRST_INPUT  # noqa: E402

FPGA_BASELINE_PATH = Path(__file__).resolve().parent / "fpga_baseline.json"


# Per LUT size: vendor flow, and how its cells count. A cell type maps to the LUT sites one
# such cell occupies.
TARGETS: dict[int, dict] = {
    6: {
        "target": "xilinx-7series",
        "flow": "synth_xilinx -flatten -noiopad",
        "lut_sites": {f"LUT{i}": 1 for i in range(1, 7)} | {"INV": 1},
        "carry": ("CARRY4",),
        "mux": ("MUXF7", "MUXF8"),
    },
    4: {
        "target": "lattice-ecp5",
        "flow": "synth_ecp5 -flatten -noiopad",
        "lut_sites": {"LUT4": 1, "CCU2C": 2},
        "carry": ("CCU2C",),
        "mux": ("PFUMX", "L6MUX21"),
    },
}


def netlist_to_verilog(netlist: Netlist) -> str:
    """Our compiled circuit as plain Verilog: one truth-table lookup per LUT, no structure."""
    n_out = len(netlist.outputs)

    def name(s: int) -> str:
        if s == CONST0:
            return "1'b0"
        if s == CONST1:
            return "1'b1"
        if s < netlist.first_lut_signal:
            return f"x[{s - FIRST_INPUT}]"
        return f"s[{s - netlist.first_lut_signal}]"

    lines = [f"module top(input [{netlist.num_inputs - 1}:0] x, output [{n_out - 1}:0] y);"]
    if netlist.luts:
        lines.append(f"  wire [{len(netlist.luts) - 1}:0] s;")
    for n, lut in enumerate(netlist.luts):
        width = 1 << len(lut.inputs)
        index = ", ".join(name(s) for s in reversed(lut.inputs))
        lines.append(f"  wire [{width - 1}:0] t{n} = {width}'h{lut.table:x};")
        lines.append(f"  assign s[{n}] = t{n}[{{{index}}}];")
    for j, s in enumerate(netlist.outputs):
        lines.append(f"  assign y[{j}] = {name(s)};")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def parse_stat(text: str) -> dict[str, int]:
    """Cell counts by type from the output of Yosys `stat`."""
    cells: dict[str, int] = {}
    for count, cell in re.findall(r"^\s+(\d+)\s+(\$?[A-Za-z_][\w$]*)\s*$", text, re.M):
        if cell not in ("wires", "cells", "ports", "processes", "memories"):
            cells[cell] = cells.get(cell, 0) + int(count)
    return cells


def summarize(cells: dict[str, int], k: int) -> dict:
    """LUT sites, carry cells and slice muxes of a mapped design on the target for k."""
    spec = TARGETS[k]
    known = set(spec["lut_sites"]) | set(spec["carry"]) | set(spec["mux"])
    unknown = sorted(set(cells) - known)
    if unknown:
        raise ValueError(f"unexpected cells for {spec['target']}: {unknown}")
    return {
        "lut_sites": sum(cells.get(c, 0) * w for c, w in spec["lut_sites"].items()),
        "carry_cells": sum(cells.get(c, 0) for c in spec["carry"]),
        "mux_cells": sum(cells.get(c, 0) for c in spec["mux"]),
        "cells": dict(sorted(cells.items())),
    }


def synthesize_verilog(verilog: str, k: int, timeout: float) -> dict:
    """Run the vendor flow for LUT size k on a `top` module; returns the summary and runtime."""
    yosys = yosys_binary()
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "net.v").write_text(verilog)
        # Relative paths and cwd: the WebAssembly build only sees its working directory.
        script = f"read_verilog net.v; {TARGETS[k]['flow']} -top top; tee -q -o stat.txt stat"
        start = time.perf_counter()
        subprocess.run(
            [yosys, "-q", "-p", script],
            check=True,
            timeout=timeout,
            cwd=tmp,
            stdout=subprocess.DEVNULL,
        )
        seconds = time.perf_counter() - start
        cells = parse_stat((Path(tmp) / "stat.txt").read_text())
    return {**summarize(cells, k), "seconds": round(seconds, 1)}


def load_fpga_baseline() -> dict:
    if not FPGA_BASELINE_PATH.exists():
        return {}
    return json.loads(FPGA_BASELINE_PATH.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", help="only these cases (default: all)")
    parser.add_argument("--timeout", type=float, default=3600, help="seconds per synthesis")
    parser.add_argument("--output", type=Path, default=FPGA_BASELINE_PATH, help="JSON to update")
    parser.add_argument(
        "--network-only", action="store_true", help="skip handing our circuit to the flow"
    )
    args = parser.parse_args()
    baseline = json.loads(args.output.read_text()) if args.output.exists() else {}
    version = subprocess.run([yosys_binary(), "-V"], capture_output=True, text=True).stdout
    for case in SUITE:
        if args.cases and case.name not in args.cases:
            continue
        net = random_network(case.layer_sizes, case.sparsity, np.random.default_rng(case.seed))
        try:
            network = synthesize_verilog(to_verilog(net), case.k, args.timeout)
        except subprocess.TimeoutExpired:
            print(f"{case.name}: Yosys did not finish within {args.timeout:.0f} s", file=sys.stderr)
            continue
        # Informational only (the benchmark does not read it), so a timeout just leaves it out.
        through = None
        if not args.network_only:
            ours = compile_network(net, k=case.k)
            try:
                through = synthesize_verilog(netlist_to_verilog(ours), case.k, args.timeout)
                through["our_lut_count"] = ours.lut_count()
            except subprocess.TimeoutExpired:
                print(f"{case.name}: our circuit timed out", file=sys.stderr)
        baseline[case.name] = {
            "target": TARGETS[case.k]["target"],
            "flow": TARGETS[case.k]["flow"],
            "tool": version.strip().split(" (")[0],
            "network": network,
            "ours_through_flow": through,
        }
        print(f"{case.name}: {json.dumps(baseline[case.name])}")
        args.output.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
