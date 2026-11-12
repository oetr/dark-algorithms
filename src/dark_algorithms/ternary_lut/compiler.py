"""Compile a ternary network into a k-LUT netlist with no weight memory.

Strategy (v1, deliberately simple so later changes have a clear baseline):

* Zero weights are dropped: a neuron only wires up its non-zero inputs.
* A neuron whose remaining fan-in fits in one LUT (<= k inputs) becomes a single
  LUT whose truth table is the threshold function itself.
* A wider neuron is split into chunks of <= k inputs. Each chunk's partial sum is
  produced as a two's-complement integer, one LUT per bit (the threshold is folded
  into the first chunk). Partial sums are added in a balanced tree of ripple-carry
  adders built from LUTs, and the neuron fires when the final sum is non-negative,
  i.e. when its sign bit is 0.
* Identical LUTs are shared (structural hashing), constant LUTs become constant
  signals, and LUTs no output depends on are swept away.

Known headroom, roughly in order of expected payoff:
  - fold the sign-bit inverter into consuming LUTs instead of spending a LUT on it,
  - wider adders per LUT level (a k=6 LUT can add two 3-bit numbers in one level),
  - grouping inputs into chunks by shared structure across neurons,
  - exact synthesis for small neurons.

Setting `prune_zeros=False` compiles every weight, zero or not, as if the
hardware could not exploit sparsity. It is the baseline the eval compares against.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product

from .model import TernaryNetwork
from .netlist import CONST0, CONST1, Lut, Netlist

DEFAULT_K = 6


def bits_for_range(lo: int, hi: int) -> int:
    """Smallest two's-complement width that represents every integer in [lo, hi]."""
    if lo > hi:
        raise ValueError("empty range")
    width = 1
    while not (-(1 << (width - 1)) <= lo and hi <= (1 << (width - 1)) - 1):
        width += 1
    return width


@dataclass(frozen=True)
class IntSignal:
    """A signed integer on `bits` (LSB first, two's complement), known to lie in [lo, hi]."""

    bits: tuple[int, ...]
    lo: int
    hi: int

    def bit(self, j: int) -> int:
        """Bit j, sign-extended past the stored width."""
        return self.bits[min(j, len(self.bits) - 1)]


class _Builder:
    def __init__(self, k: int, num_inputs: int) -> None:
        self.netlist = Netlist(k=k, num_inputs=num_inputs)
        self._cache: dict[tuple[tuple[int, ...], int], int] = {}

    def lut(self, inputs: Sequence[int], fn: Callable[[tuple[int, ...]], int]) -> int:
        """Add a LUT computing fn over the given input signals; returns its output signal."""
        inputs = tuple(inputs)
        if len(inputs) > self.netlist.k:
            raise ValueError(f"{len(inputs)} inputs exceed k={self.netlist.k}")
        table = 0
        for idx, bits in enumerate(product((0, 1), repeat=len(inputs))):
            # product() varies the last element fastest; reverse so inputs[0] is the LSB.
            bits = tuple(reversed(bits))
            if fn(bits) & 1:
                table |= 1 << idx
        full = (1 << (1 << len(inputs))) - 1
        if table == 0:
            return CONST0
        if table == full:
            return CONST1
        key = (inputs, table)
        if key not in self._cache:
            self.netlist.luts.append(Lut(inputs, table))
            self._cache[key] = self.netlist.num_signals - 1
        return self._cache[key]

    # -- arithmetic -------------------------------------------------------------------------

    def chunk_sum(self, terms: Sequence[tuple[int, int]], offset: int) -> IntSignal:
        """Partial sum offset + sum(w * x) over <= k (signal, weight) terms, one LUT per bit."""
        signals = [s for s, _ in terms]
        weights = [w for _, w in terms]
        lo = offset + sum(min(w, 0) for w in weights)
        hi = offset + sum(max(w, 0) for w in weights)
        width = bits_for_range(lo, hi)

        def value(bits: tuple[int, ...]) -> int:
            return offset + sum(w * b for w, b in zip(weights, bits, strict=True))

        out = tuple(
            self.lut(signals, lambda bits, j=j: (value(bits) >> j) & 1) for j in range(width)
        )
        return IntSignal(out, lo, hi)

    def add(self, a: IntSignal, b: IntSignal) -> IntSignal:
        """Ripple-carry adder from LUTs. The result width covers the exact sum range."""
        lo, hi = a.lo + b.lo, a.hi + b.hi
        width = bits_for_range(lo, hi)
        out: list[int] = []
        carry: int | None = None
        for j in range(width):
            x, y = a.bit(j), b.bit(j)
            last = j == width - 1
            if carry is None:
                out.append(self.lut((x, y), lambda v: v[0] ^ v[1]))
                if not last:
                    carry = self.lut((x, y), lambda v: v[0] & v[1])
            else:
                out.append(self.lut((x, y, carry), lambda v: v[0] ^ v[1] ^ v[2]))
                if not last:
                    carry = self.lut(
                        (x, y, carry), lambda v: (v[0] & v[1]) | (v[2] & (v[0] ^ v[1]))
                    )
        return IntSignal(tuple(out), lo, hi)

    # -- neurons ----------------------------------------------------------------------------

    def neuron(self, terms: Sequence[tuple[int, int]], threshold: int) -> int:
        """Signal that is 1 iff sum(w * x) >= threshold."""
        k = self.netlist.k
        if len(terms) <= k:
            signals = [s for s, _ in terms]
            weights = [w for _, w in terms]
            return self.lut(
                signals,
                lambda bits: int(
                    sum(w * b for w, b in zip(weights, bits, strict=True)) >= threshold
                ),
            )
        chunks = [terms[i : i + k] for i in range(0, len(terms), k)]
        partials = [self.chunk_sum(c, -threshold if i == 0 else 0) for i, c in enumerate(chunks)]
        while len(partials) > 1:
            nxt = [self.add(partials[i], partials[i + 1]) for i in range(0, len(partials) - 1, 2)]
            if len(partials) % 2:
                nxt.append(partials[-1])
            partials = nxt
        total = partials[0]
        if total.lo >= 0:
            return CONST1
        if total.hi < 0:
            return CONST0
        sign = total.bits[-1]
        return self.lut((sign,), lambda v: 1 - v[0])


def compile_network(
    network: TernaryNetwork, k: int = DEFAULT_K, prune_zeros: bool = True
) -> Netlist:
    """Compile `network` into a validated, swept k-LUT netlist with identical behaviour."""
    if k < 3:
        raise ValueError("k must be at least 3 (the adder needs 3-input LUTs)")
    b = _Builder(k, network.num_inputs)
    current = [b.netlist.input_signal(i) for i in range(network.num_inputs)]
    for layer in network.layers:
        nxt = []
        for j in range(layer.out_features):
            row = layer.weights[j]
            terms = [
                (current[i], int(row[i]))
                for i in range(layer.in_features)
                if row[i] != 0 or not prune_zeros
            ]
            nxt.append(b.neuron(terms, int(layer.thresholds[j])))
        current = nxt
    b.netlist.outputs = current
    netlist = b.netlist.sweep()
    netlist.validate()
    return netlist
