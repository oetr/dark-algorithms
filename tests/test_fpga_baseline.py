"""The real-FPGA reference is only meaningful if it counts the right cells and, when it is
handed our circuit, the Verilog it gets computes that circuit."""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from fpga_baseline import (  # noqa: E402
    TARGETS,
    load_fpga_baseline,
    netlist_to_verilog,
    parse_stat,
    summarize,
    synthesize_verilog,
)
from suite import SUITE  # noqa: E402
from yosys_baseline import (  # noqa: E402
    simulate_yosys_netlist,
    synthesize_verilog_netlist,
    to_verilog,
)

from dark_algorithms.ternary_lut import Lut, Netlist, compile_network, random_network  # noqa: E402

HAVE_YOSYS = bool(shutil.which("yowasp-yosys") or shutil.which("yosys"))

XILINX_STAT = """
=== top ===

        +----------Local Count, excluding submodules.
        |
      336 wires
      518 wire bits
        2 public wires
       24 public wire bits
        2 ports
       24 port bits
      174 cells
       35   CARRY4
        3   INV
       51   LUT2
       34   LUT3
       19   LUT4
       18   LUT5
       12   LUT6
        2   MUXF7
"""


def test_parse_stat_reads_cell_counts_only():
    cells = parse_stat(XILINX_STAT)
    assert cells == {
        "CARRY4": 35,
        "INV": 3,
        "LUT2": 51,
        "LUT3": 34,
        "LUT4": 19,
        "LUT5": 18,
        "LUT6": 12,
        "MUXF7": 2,
    }


def test_xilinx_lut_sites_count_luts_and_inverters_not_carry_or_muxes():
    s = summarize(parse_stat(XILINX_STAT), k=6)
    assert s["lut_sites"] == 3 + 51 + 34 + 19 + 18 + 12
    assert s["carry_cells"] == 35
    assert s["mux_cells"] == 2


def test_ecp5_carry_cell_occupies_two_lut4_sites():
    s = summarize({"LUT4": 10, "CCU2C": 4, "PFUMX": 3}, k=4)
    assert s["lut_sites"] == 10 + 2 * 4
    assert s["carry_cells"] == 4
    assert s["mux_cells"] == 3


def test_unknown_cells_are_rejected_rather_than_silently_dropped():
    with pytest.raises(ValueError, match="DSP48E1"):
        summarize({"LUT6": 1, "DSP48E1": 1}, k=6)


def test_every_suite_lut_size_has_a_target():
    assert {case.k for case in SUITE} <= set(TARGETS)


def test_stored_baseline_matches_the_targets():
    for name, entry in load_fpga_baseline().items():
        case = next(c for c in SUITE if c.name == name)
        assert entry["target"] == TARGETS[case.k]["target"]
        assert entry["network"]["lut_sites"] > 0


def test_netlist_verilog_names_constants_inputs_and_luts():
    nl = Netlist(k=3, num_inputs=2, luts=[Lut((2, 3), 0b1000), Lut((4, 1), 0b0110)])
    nl.outputs = [5, 0, 3]
    v = netlist_to_verilog(nl)
    assert "wire [3:0] t0 = 4'h8;" in v
    assert "assign s[0] = t0[{x[1], x[0]}];" in v
    assert "assign s[1] = t1[{1'b1, s[0]}];" in v
    assert "assign y[0] = s[1];" in v
    assert "assign y[1] = 1'b0;" in v
    assert "assign y[2] = x[1];" in v


@pytest.mark.skipif(not HAVE_YOSYS, reason="Yosys not installed")
@pytest.mark.parametrize("k", [4, 6])
def test_exported_circuit_matches_reference_exhaustively(k):
    net = random_network([12, 10, 5], 0.4, np.random.default_rng(k))
    ours = compile_network(net, k=k)
    mapped = synthesize_verilog_netlist(netlist_to_verilog(ours), k=6)
    idx = np.arange(1 << 12)
    x = ((idx[:, None] >> np.arange(12)) & 1).astype(np.uint8)
    assert (simulate_yosys_netlist(mapped, x) == net.forward(x)).all()


@pytest.mark.skipif(not HAVE_YOSYS, reason="Yosys not installed")
def test_vendor_flow_uses_carry_chains_on_the_network():
    net = random_network([16, 8, 4], 0.5, np.random.default_rng(0))
    result = synthesize_verilog(to_verilog(net), k=4, timeout=300)
    assert result["carry_cells"] > 0
    assert result["lut_sites"] >= 2 * result["carry_cells"]
