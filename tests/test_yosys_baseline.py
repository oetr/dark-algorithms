"""The Yosys baseline is only fair if the Verilog it is given computes the same network."""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from yosys_baseline import simulate_yosys_netlist, synthesize_netlist, to_verilog  # noqa: E402

from dark_algorithms.ternary_lut import random_network  # noqa: E402


def test_verilog_has_one_assignment_per_neuron():
    net = random_network([12, 8, 4], 0.5, np.random.default_rng(0))
    assert to_verilog(net).count("assign ") == 8 + 4


@pytest.mark.skipif(
    not (shutil.which("yowasp-yosys") or shutil.which("yosys")), reason="Yosys not installed"
)
@pytest.mark.parametrize("seed", range(3))
def test_yosys_netlist_matches_reference_exhaustively(seed):
    net = random_network([12, 10, 5], 0.4, np.random.default_rng(seed))
    netlist = synthesize_netlist(net, k=6)
    idx = np.arange(1 << 12)
    x = ((idx[:, None] >> np.arange(12)) & 1).astype(np.uint8)
    assert (simulate_yosys_netlist(netlist, x) == net.forward(x)).all()
