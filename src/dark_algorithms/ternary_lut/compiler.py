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
from dataclasses import dataclass, field
from functools import cache

import numpy as np

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


@cache
def combinations(n: int) -> np.ndarray:
    """All 2^n input combinations as rows of bits; row idx has bit j = (idx >> j) & 1."""
    idx = np.arange(1 << n, dtype=np.int64)
    out = ((idx[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.int64)
    out.flags.writeable = False
    return out


def elem_values(elems: Sequence[tuple[Elem, int]], first_column: int = 0) -> np.ndarray:
    """Value of sum(e.value(bit) << shift) for every combination of the elems' signals.

    The elems occupy LUT inputs first_column.. of a LUT with first_column + len(elems) inputs.
    """
    n = first_column + len(elems)
    rows = combinations(n)[:, first_column:]
    lo = np.array([e.lo << sh for e, sh in elems], dtype=np.int64)
    hi = np.array([e.hi << sh for e, sh in elems], dtype=np.int64)
    return lo.sum() + rows @ (hi - lo)


@dataclass(frozen=True)
class Elem:
    """A heap bit: `signal` read as the value lo (signal 0) or hi (signal 1), each 0 or 1.

    (lo, hi) = (0, 1) is the signal itself, (1, 0) its complement, which is how a -1 weight
    enters a popcount for free (the inversion folds into the truth table of the LUT that
    reads it), and (0, 0) a zero weight kept only by the dense baseline.
    """

    signal: int
    lo: int = 0
    hi: int = 1

    def value(self, bit: int) -> int:
        return self.hi if bit else self.lo

    @property
    def hi_value(self) -> int:
        return max(self.lo, self.hi)


@dataclass
class BitHeap:
    """Columns of weighted bits: value = offset + sum over columns c of 2^c * sum(bits in c)."""

    cols: dict[int, list[Elem]] = field(default_factory=dict)
    offset: int = 0

    def extend(self, col: int, elems: Sequence[Elem]) -> None:
        self.cols.setdefault(col, []).extend(elems)

    def add_signal(self, col: int, signal: int) -> None:
        if signal == CONST1:
            self.offset += 1 << col
        elif signal != CONST0:
            self.cols.setdefault(col, []).append(Elem(signal))

    def max_height(self) -> int:
        return max((len(v) for v in self.cols.values()), default=0)


class _Builder:
    def __init__(self, k: int, num_inputs: int) -> None:
        self.netlist = Netlist(k=k, num_inputs=num_inputs)
        self._cache: dict[tuple[tuple[int, ...], int], int] = {}
        self._level = [0] * self.netlist.first_lut_signal  # logic depth of every signal
        self._added: list[tuple[tuple[int, ...], int]] = []  # cache keys in creation order

    def lut(self, inputs: Sequence[int], fn: Callable[[tuple[int, ...]], int]) -> int:
        """Add a LUT computing fn over the given input signals; returns its output signal."""
        inputs = tuple(inputs)
        if len(inputs) > self.netlist.k:
            raise ValueError(f"{len(inputs)} inputs exceed k={self.netlist.k}")
        truth = [fn(tuple(int(b) for b in row)) & 1 for row in combinations(len(inputs))]
        return self.lut_from_truth(inputs, np.array(truth, dtype=bool))

    def lut_from_truth(self, inputs: Sequence[int], truth: np.ndarray) -> int:
        """Add a LUT whose output for input combination idx is truth[idx] (inputs[0] = LSB)."""
        inputs = tuple(inputs)
        if len(inputs) > self.netlist.k:
            raise ValueError(f"{len(inputs)} inputs exceed k={self.netlist.k}")
        table = int.from_bytes(np.packbits(truth, bitorder="little").tobytes(), "little")
        full = (1 << (1 << len(inputs))) - 1
        if table == 0:
            return CONST0
        if table == full:
            return CONST1
        key = (inputs, table)
        if key not in self._cache:
            self.netlist.luts.append(Lut(inputs, table))
            self._cache[key] = self.netlist.num_signals - 1
            self._added.append(key)
            self._level.append(1 + max(self._level[s] for s in inputs))
        return self._cache[key]

    def level(self, signal: int) -> int:
        return self._level[signal]

    def trial(self, build: Callable[[], int]) -> tuple[int, int]:
        """Run `build`, report (new LUTs, depth of its output), then undo every LUT it added."""
        mark = len(self.netlist.luts)
        out = build()
        cost = (len(self.netlist.luts) - mark, self._level[out])
        first_new = self.netlist.first_lut_signal + mark
        del self.netlist.luts[mark:]
        del self._level[first_new:]
        for key in self._added[mark:]:
            del self._cache[key]
        del self._added[mark:]
        return cost

    # -- popcount threshold: compressor tree + comparator chain ------------------------------

    def counter(self, elems: Sequence[Elem]) -> list[tuple[int, int]]:
        """Generalised counter: the binary sum of `elems` (all of one column), one LUT per bit.

        Returns (signal, relative column) pairs; constant output bits come back as
        CONST0/CONST1 signals and are dealt with by the caller.
        """
        signals = [e.signal for e in elems]
        top = sum(e.hi_value for e in elems)
        if top == 0:
            return []

        values = elem_values([(e, 0) for e in elems])
        return [
            (self.lut_from_truth(signals, (values >> j) & 1 == 1), j)
            for j in range(top.bit_length())
        ]

    def compress(self, heap: BitHeap, max_height: int) -> None:
        """Wallace-style reduction with k-input counters until no column exceeds max_height.

        Every stage adds one LUT level. Within a stage each column is cut into groups of up
        to k bits; a group of g bits becomes ceil(log2(g + 1)) bits spread over the next
        columns, so a k=6 group (a 6:3 counter) removes three bits for three LUTs.
        """
        k = self.netlist.k
        # Groups of 1-2 bits remove nothing; only a target height of 1 needs half adders.
        min_group = 3 if max_height >= 2 else 2
        while heap.max_height() > max_height:
            nxt = BitHeap(offset=heap.offset)
            for col in sorted(heap.cols):
                elems = heap.cols[col]
                if len(elems) <= max_height:
                    nxt.extend(col, elems)
                    continue
                for i in range(0, len(elems), k):
                    group = elems[i : i + k]
                    if len(group) < min_group:
                        nxt.extend(col, group)
                        continue
                    for sig, j in self.counter(group):
                        nxt.add_signal(col + j, sig)
            heap.cols, heap.offset = nxt.cols, nxt.offset

    def heap_at_least(self, heap: BitHeap, threshold: int) -> int:
        """Signal that is 1 iff the heap's value is >= threshold.

        Columns are consumed from the least significant one upwards. The state passed from
        one step to the next is the integer carry q = floor((low part - low threshold bits)
        / 2^c), kept as an offset-binary number (often one bit, sometimes a constant). Each
        step spends one LUT per carry bit over the carry and as many whole columns as fit
        into k inputs; the last step compares directly and yields the neuron output.
        """
        k = self.netlist.k
        threshold -= heap.offset
        c = 0
        carry: list[tuple[int, int]] = []  # (signal, bit position) of q - q_lo
        q_lo = 0
        while True:
            cols = [col for col in sorted(heap.cols) if col >= c and heap.cols[col]]
            t_rest = threshold >> c  # floor division, also for negative thresholds
            q_hi = q_lo + sum(1 << j for _, j in carry)
            hi_rest = sum(e.hi_value << (col - c) for col in cols for e in heap.cols[col])
            if q_lo >= t_rest:
                return CONST1
            if hi_rest + q_hi < t_rest:
                return CONST0
            take: list[int] = []
            used = len(carry)
            for col in cols:
                if used + len(heap.cols[col]) > k:
                    break
                take.append(col)
                used += len(heap.cols[col])
            if not take:
                # Column c plus the carry overflow one LUT (only for small k): compress the
                # unconsumed columns further. Their constants are multiples of 2^c, so moving
                # them into the threshold leaves the carry computed so far valid.
                rest = BitHeap(cols={col: heap.cols.pop(col) for col in cols})
                self.compress(rest, max_height=max(1, k - len(carry)))
                heap.cols.update(rest.cols)
                threshold -= rest.offset
                continue
            elems = [(e, col - c) for col in take for e in heap.cols[col]]
            signals = [s for s, _ in carry] + [e.signal for e, _ in elems]

            n = len(carry)
            values = q_lo + elem_values(elems, first_column=n)
            if n:
                weights = np.array([1 << j for _, j in carry], dtype=np.int64)
                values = values + combinations(n + len(elems))[:, :n] @ weights

            if len(take) == len(cols):
                return self.lut_from_truth(signals, values >= t_rest)
            delta = take[-1] + 1 - c
            t_low = t_rest & ((1 << delta) - 1)
            v_hi = sum(e.hi_value << sh for e, sh in elems)
            nq_lo = (q_lo - t_low) >> delta
            nq_hi = (q_hi + v_hi - t_low) >> delta
            next_q = ((values - t_low) >> delta) - nq_lo
            new_carry: list[tuple[int, int]] = []
            new_lo = nq_lo
            for j in range((nq_hi - nq_lo).bit_length()):
                sig = self.lut_from_truth(signals, (next_q >> j) & 1 == 1)
                if sig == CONST1:
                    new_lo += 1 << j
                elif sig != CONST0:
                    new_carry.append((sig, j))
            carry, q_lo, c = new_carry, new_lo, c + delta

    # -- neurons ----------------------------------------------------------------------------

    def neuron(self, terms: Sequence[tuple[int, int]], threshold: int) -> int:
        """Signal that is 1 iff sum(w * x) >= threshold.

        With x' = 1 - x for every -1 weight, sum(w * x) = popcount(literals) - #negative, so
        the neuron is a popcount threshold over literals whose inversions cost nothing.
        """
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
        elems = [Elem(s, 1, 0) if w < 0 else Elem(s, 0, 1 if w > 0 else 0) for s, w in terms]
        literal_threshold = threshold + sum(1 for _, w in terms if w < 0)

        def build(max_height: int) -> int:
            heap = BitHeap(cols={0: list(elems)})
            self.compress(heap, max_height=max_height)
            return self.heap_at_least(heap, literal_threshold)

        # Tall columns mean fewer counter LUTs but a longer comparator chain; which height
        # is cheapest depends on the fan-in, so try them all and keep the best.
        best = min(range(2, k + 1), key=lambda h: (*self.trial(lambda: build(h)), h))
        return build(best)


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
