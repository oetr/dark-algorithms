"""LUT netlists: the *output* of the compiler, i.e. what we actually produce.

A netlist is a directed acyclic graph of k-input lookup tables, the basic logic
element of an FPGA. It holds no memory cells: every weight of the source network
has been folded into LUT truth tables.

Signals are integer ids:
  * 0 and 1 are the constants false and true,
  * 2 .. 2 + num_inputs - 1 are the primary inputs,
  * every LUT drives exactly one new signal, numbered in creation order.

LUTs are stored in topological order, so a LUT only reads signals created before it.

A truth table is an int bitmask: bit `idx` holds the output for the input
combination `idx = sum(value(inputs[j]) << j)`.

`layer_outputs` optionally records, for every layer of the source network, the signal that
carries each neuron's output (None when no output depends on it and it was swept away).
It adds no logic; the formal equivalence proof uses it to check one layer at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CONST0 = 0
CONST1 = 1
FIRST_INPUT = 2


@dataclass(frozen=True)
class Lut:
    inputs: tuple[int, ...]
    table: int

    def evaluate(self, bits: tuple[int, ...]) -> int:
        idx = sum(b << j for j, b in enumerate(bits))
        return (self.table >> idx) & 1


@dataclass
class Netlist:
    k: int
    num_inputs: int
    luts: list[Lut] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    layer_outputs: list[list[int | None]] = field(default_factory=list)

    @property
    def first_lut_signal(self) -> int:
        return FIRST_INPUT + self.num_inputs

    @property
    def num_signals(self) -> int:
        return self.first_lut_signal + len(self.luts)

    def input_signal(self, i: int) -> int:
        if not 0 <= i < self.num_inputs:
            raise IndexError(i)
        return FIRST_INPUT + i

    def lut_for_signal(self, signal: int) -> Lut | None:
        if signal < self.first_lut_signal:
            return None
        return self.luts[signal - self.first_lut_signal]

    def validate(self) -> None:
        """Raise ValueError if the netlist is not a well-formed k-LUT DAG."""
        for n, lut in enumerate(self.luts):
            own = self.first_lut_signal + n
            if len(lut.inputs) > self.k:
                raise ValueError(f"LUT {own} has {len(lut.inputs)} inputs, more than k={self.k}")
            if lut.table >> (1 << len(lut.inputs)):
                raise ValueError(f"LUT {own} truth table is wider than 2^{len(lut.inputs)} bits")
            for s in lut.inputs:
                if not 0 <= s < own:
                    raise ValueError(f"LUT {own} reads signal {s}, which is not defined before it")
        for s in self.outputs:
            if not 0 <= s < self.num_signals:
                raise ValueError(f"output refers to undefined signal {s}")
        for layer in self.layer_outputs:
            for s in layer:
                if s is not None and not 0 <= s < self.num_signals:
                    raise ValueError(f"layer output refers to undefined signal {s}")

    def depth(self) -> int:
        """Logic depth: the longest chain of LUTs from an input to an output.

        This is the high-level proxy for the circuit's critical path (clock speed).
        """
        level = [0] * self.num_signals
        for n, lut in enumerate(self.luts):
            level[self.first_lut_signal + n] = 1 + max((level[s] for s in lut.inputs), default=0)
        return max((level[s] for s in self.outputs), default=0)

    def lut_count(self) -> int:
        return len(self.luts)

    def lut_input_histogram(self) -> dict[int, int]:
        hist: dict[int, int] = {}
        for lut in self.luts:
            hist[len(lut.inputs)] = hist.get(len(lut.inputs), 0) + 1
        return dict(sorted(hist.items()))

    def sweep(self) -> Netlist:
        """Return a copy without LUTs that no output depends on."""
        live = set(self.outputs)
        for n in range(len(self.luts) - 1, -1, -1):
            sig = self.first_lut_signal + n
            if sig in live:
                live.update(self.luts[n].inputs)
        remap = {s: s for s in range(self.first_lut_signal)}
        kept: list[Lut] = []
        for n, lut in enumerate(self.luts):
            sig = self.first_lut_signal + n
            if sig in live:
                remap[sig] = self.first_lut_signal + len(kept)
                kept.append(Lut(tuple(remap[s] for s in lut.inputs), lut.table))
        return Netlist(
            k=self.k,
            num_inputs=self.num_inputs,
            luts=kept,
            outputs=[remap[s] for s in self.outputs],
            layer_outputs=[[remap.get(s) for s in layer] for layer in self.layer_outputs],
        )
