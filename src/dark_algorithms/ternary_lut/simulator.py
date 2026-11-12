"""High-level functional simulator for LUT netlists.

Simulates at the LUT level, one evaluation per LUT for a whole batch of input
vectors at once (vectorised with NumPy). There is no gate-level or timing
simulation: correctness is checked functionally, and timing is approximated by
the netlist's logic depth. That keeps checking a candidate circuit to seconds
rather than the hours an FPGA toolchain would take.
"""

from __future__ import annotations

import numpy as np

from .netlist import CONST1, FIRST_INPUT, Netlist


def _table_array(table: int, num_inputs: int) -> np.ndarray:
    return np.array([(table >> i) & 1 for i in range(1 << num_inputs)], dtype=np.uint8)


def simulate(netlist: Netlist, inputs: np.ndarray) -> np.ndarray:
    """Evaluate the netlist on a batch of bit vectors.

    inputs: array of shape (batch, num_inputs) with values in {0, 1}.
    Returns uint8 array of shape (batch, num_outputs).
    """
    x = np.asarray(inputs, dtype=np.uint8)
    if x.ndim != 2 or x.shape[1] != netlist.num_inputs:
        raise ValueError(f"inputs must have shape (batch, {netlist.num_inputs}), got {x.shape}")
    batch = x.shape[0]
    values = np.zeros((netlist.num_signals, batch), dtype=np.uint8)
    values[CONST1] = 1
    values[FIRST_INPUT : FIRST_INPUT + netlist.num_inputs] = x.T
    base = netlist.first_lut_signal
    for n, lut in enumerate(netlist.luts):
        idx = np.zeros(batch, dtype=np.int64)
        for j, s in enumerate(lut.inputs):
            idx |= values[s].astype(np.int64) << j
        values[base + n] = _table_array(lut.table, len(lut.inputs))[idx]
    return values[netlist.outputs].T.copy()
