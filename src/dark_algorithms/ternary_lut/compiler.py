"""Compile a ternary network into a k-LUT netlist with no weight memory.

Strategy:

* Zero weights are dropped: a neuron only wires up its non-zero inputs.
* A neuron whose remaining fan-in fits in one LUT (<= k inputs) becomes a single
  LUT whose truth table is the threshold function itself.
* A wider neuron is rewritten as a popcount threshold: with x' = 1 - x for every -1
  weight, sum(w * x) >= t  <=>  popcount(literals) >= t + #negative weights. The
  inversions fold into the truth tables of the LUTs that read the literals, so they
  cost nothing.
* The literals form a bit heap (columns of equally weighted bits) that a Wallace-style
  tree of generalised counters compresses: each LUT level cuts every column into groups
  of up to k bits, and a group of g bits becomes ceil(log2(g + 1)) bits in the next
  columns (a 6:3 counter removes three bits for three LUTs, where a ripple-carry adder
  removes about one bit per two LUTs).
* The compressed heap is compared with the constant threshold by a comparator chain
  that walks the columns from the least significant one and passes on only the small
  integer carry floor((low part - low threshold bits) / 2^c). Each step reads as many
  whole columns as fit next to the carry, and the last step is the neuron's output, so
  no adder result or sign bit is ever materialised.
* Tall columns mean fewer counters but a longer chain, so every neuron is built for each
  target column height 2..k and the cheapest version (fewest new LUTs, then least depth)
  is kept.
* A second candidate per neuron adds 2^W - T' so the threshold sits on a power of two
  and only bit W of the sum modulo 2^(W+1) is needed. Its heap is compressed by
  two-column generalised parallel counters ((6;3), (1,5;3), (2,3;3), full adders at k=4)
  until columns hold <= 2 bits, and a carry chain's last LUT emits bit W. It is kept
  when it has fewer LUTs and no more depth than the best counter tree (mostly at k=4).
* Before any of this, constant inputs are folded into the threshold and a signal that
  feeds a neuron twice is merged into its net weight, so x and -x cancel.
* Identical LUTs are shared (structural hashing), constant LUTs become constant
  signals, 1-input buffers are wired through, and LUTs no output depends on are swept.

Known headroom, roughly in order of expected payoff:
  - two-column GPCs inside the counter tree's stages rather than as a separate build,
    and a Dadda-style schedule that compresses only as much as each column needs,
  - network-level depth budgeting: the GPC build saves another ~1% of LUTs when allowed
    one extra level on neurons that are off the critical path,
  - sharing literal groups (counters) across neurons that read the same inputs,
  - exact synthesis for small cones,
  - choosing per-neuron structures against the network's critical path instead of
    neuron-local depth.

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


@cache
def combinations(n: int) -> np.ndarray:
    """All 2^n input combinations as rows of bits; row idx has bit j = (idx >> j) & 1."""
    idx = np.arange(1 << n, dtype=np.int64)
    out = ((idx[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.int64)
    out.flags.writeable = False
    return out


def elem_values(elems: Sequence[tuple[Elem, int]], first_column: int = 0) -> np.ndarray:
    """Value of sum(bit-read-as-lo-or-hi << shift) for every combination of the elems' signals.

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
        if len(inputs) == 1 and table == 0b10:
            return inputs[0]  # a buffer: wire the input through
        key = (inputs, table)
        if key not in self._cache:
            self.netlist.luts.append(Lut(inputs, table))
            self._cache[key] = self.netlist.num_signals - 1
            self._added.append(key)
            self._level.append(1 + max(self._level[s] for s in inputs))
        return self._cache[key]

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

    def lut_lits(self, lits: Sequence[Lit], fn: Callable[[tuple[int, ...]], int]) -> int:
        """LUT over literals: constants are folded and inversions absorbed into the table.

        fn receives the literal values (after inversion) in the order of `lits`.
        """
        fixed: dict[int, int] = {}
        signals: list[int] = []
        slot: list[int] = []
        for pos, (s, inv, _) in enumerate(lits):
            if s in (CONST0, CONST1):
                fixed[pos] = int(s == CONST1) ^ int(inv)
            else:
                slot.append(pos)
                signals.append(s)

        def table_fn(bits: tuple[int, ...]) -> int:
            values = [0] * len(lits)
            for pos, v in fixed.items():
                values[pos] = v
            for pos, b in zip(slot, bits, strict=True):
                values[pos] = b ^ int(lits[pos][1])
            return fn(tuple(values))

        if not signals:
            return CONST1 if table_fn(()) & 1 else CONST0
        return self.lut(signals, table_fn)

    def neuron(self, terms: Sequence[tuple[int, int]], threshold: int) -> int:
        """Signal that is 1 iff sum(w * x) >= threshold.

        With x' = 1 - x for every -1 weight, sum(w * x) = popcount(literals) - #negative, so
        the neuron is a popcount threshold over literals whose inversions cost nothing.
        """
        k = self.netlist.k
        lits, target = _to_popcount(terms, threshold)
        hi = sum(w for _, _, w in lits)
        if target <= 0:
            return CONST1
        if target > hi:
            return CONST0
        if len(lits) <= k:
            return self.lut_lits(
                lits,
                lambda v: int(sum(w * b for (_, _, w), b in zip(lits, v, strict=True)) >= target),
            )
        elems = [
            Elem(s, 0, 0) if w == 0 else Elem(s, 1, 0) if inv else Elem(s) for s, inv, w in lits
        ]

        def tree(max_height: int) -> int:
            heap = BitHeap(cols={0: list(elems)})
            self.compress(heap, max_height=max_height)
            return self.heap_at_least(heap, target)

        def gpc() -> int:
            top = 0  # smallest W with 2^W >= target and 2^W > hi - target
            while (1 << top) < target or (1 << top) <= hi - target:
                top += 1
            return _GpcHeap(self, lits, offset=(1 << top) - target, top=top).bit_top()

        # Tall columns mean fewer counter LUTs but a longer comparator chain; which height
        # is cheapest depends on the fan-in, so try them all and keep the best. Two-column
        # GPCs beat single-column counters for some fan-ins and k (mostly k=4); use them
        # when they save LUTs without making the neuron deeper.
        costs = {h: self.trial(lambda h=h: tree(h)) for h in range(2, k + 1)}
        best = min(costs, key=lambda h: (*costs[h], h))
        gpc_luts, gpc_depth = self.trial(gpc)
        if gpc_luts < costs[best][0] and gpc_depth <= costs[best][1]:
            return gpc()
        return tree(best)


# A literal: (signal, inverted, weight). Weight 0 only appears in the dense baseline, where a
# zero weight still occupies a wire.
Lit = tuple[int, bool, int]


def _to_popcount(terms: Sequence[tuple[int, int]], threshold: int) -> tuple[list[Lit], int]:
    """Rewrite sum(w * x) >= threshold as sum(weight * literal) >= target, weights 0 or 1.

    -x == (NOT x) - 1, so a -1 weight becomes an inverted literal and raises the target by one.
    Constant inputs are folded into the target. With zero weights pruned, a signal used
    several times is merged into its net weight first, so x and -x cancel.
    """
    target = threshold
    dense = any(w == 0 for _, w in terms)
    if dense:
        merged = list(terms)
    else:
        net: dict[int, int] = {}
        for s, w in terms:
            net[s] = net.get(s, 0) + w
        merged = [(s, w) for s, w in net.items() if w != 0]
    lits: list[Lit] = []
    for s, w in merged:
        copies = abs(w) if w != 0 else 1
        inv = w < 0
        weight = 0 if w == 0 else 1
        target += copies * weight if inv else 0
        for _ in range(copies):
            if s in (CONST0, CONST1):
                target -= weight * (int(s == CONST1) ^ int(inv))
            else:
                lits.append((s, inv, weight))
    return lits, target


class _GpcHeap:
    """Bit heap compressed by two-column GPCs, summed modulo 2^(top+1) to produce bit `top`.

    Used as one more candidate per neuron next to the counter tree + comparator chain.
    """

    def __init__(self, b: _Builder, lits: Sequence[Lit], offset: int, top: int) -> None:
        self.b = b
        self.k = b.netlist.k
        self.top = top
        self.mod = 1 << (top + 1)
        self.cols: list[list[Lit]] = [[] for _ in range(top + 1)]
        self.cols[0] = list(lits)
        self.const = offset % self.mod

    def _const_bit(self, c: int) -> int:
        return (self.const >> c) & 1

    def _height(self, c: int) -> int:
        return len(self.cols[c]) + self._const_bit(c)

    def _push(self, nxt: list[list[Lit]], c: int, signal: int) -> None:
        if c > self.top:
            return
        if signal == CONST1:
            self.const = (self.const + (1 << c)) % self.mod
        elif signal != CONST0:
            nxt[c].append((signal, False, 1))

    def _counter(self, nxt: list[list[Lit]], base: int, ins: list[Lit], offs: list[int]) -> None:
        """One GPC: every output bit of sum(weight * lit << off) (+ absorbed constant) is a LUT."""
        absorb = 0
        max_val = sum(w << o for (_, _, w), o in zip(ins, offs, strict=True))
        if (
            self._const_bit(base)
            and max_val > 0
            and (max_val + 1).bit_length() == max(max_val.bit_length(), 1)
        ):
            absorb = 1
            self.const -= 1 << base
        max_val += absorb
        for j in range(max(max_val.bit_length(), 1)):
            if base + j > self.top:
                break
            sig = self.b.lut_lits(
                ins,
                lambda v, j=j: (
                    (
                        (
                            absorb
                            + sum(w * x << o for (_, _, w), x, o in zip(ins, v, offs, strict=True))
                        )
                        >> j
                    )
                    & 1
                ),
            )
            self._push(nxt, base + j, sig)

    def _best_shape(self, here: int, above: int, const: int) -> tuple[int, int]:
        """(bits from this column, bits from the next) of the most efficient fitting GPC.

        Efficiency is heap bits removed per LUT spent; ties go to the larger counter.
        """
        best_key: tuple[float, int, int] | None = None
        best = (0, 0)
        for a in range(min(above, self.k - 1) + 1):
            for h in range(1, min(here, self.k - a) + 1):
                outs = max((h + 2 * a).bit_length(), 1)
                absorbed = int(bool(const) and (h + 2 * a + 1).bit_length() == outs)
                gain = h + a + absorbed - outs
                if gain <= 0:
                    continue
                key = (gain / outs, h + a, h)
                if best_key is None or key > best_key:
                    best_key, best = key, (h, a)
        return best

    def compress(self) -> None:
        for _ in range(64):
            if all(self._height(c) <= 2 for c in range(self.top)):
                return
            nxt: list[list[Lit]] = [[] for _ in range(self.top + 1)]
            for c in range(self.top):
                pool = self.cols[c]
                while len(pool) + self._const_bit(c) >= 3:
                    above = len(self.cols[c + 1]) if c + 1 < self.top else 0
                    h, a = self._best_shape(len(pool), above, self._const_bit(c))
                    if h + a == 0:
                        break
                    ins = pool[:h] + self.cols[c + 1][:a]
                    offs = [0] * h + [1] * a
                    del pool[:h]
                    del self.cols[c + 1][:a]
                    self._counter(nxt, c, ins, offs)
                nxt[c].extend(pool)
                self.cols[c] = []
            nxt[self.top].extend(self.cols[self.top])
            self.cols = nxt
        raise RuntimeError("bit heap compression did not converge")

    def bit_top(self) -> int:
        self.compress()
        # Only the parity of the top column matters: fold it to at most one bit.
        top_col = self.cols[self.top]
        while len(top_col) > 1:
            group, top_col = top_col[: self.k], top_col[self.k :]
            sig = self.b.lut_lits(
                group, lambda v, g=group: sum(w * x for (_, _, w), x in zip(g, v, strict=True)) & 1
            )
            if sig == CONST1:
                self.const ^= 1 << self.top
            elif sig != CONST0:
                top_col.append((sig, False, 1))
        self.cols[self.top] = top_col
        # Carry chain: each LUT takes the carry in plus as many whole columns as fit in k inputs.
        carry: Lit | None = None
        c = 0
        while True:
            ins: list[Lit] = [carry] if carry is not None else []
            offs: list[int] = [0] if carry is not None else []
            start = c
            end = c
            while end <= self.top and len(ins) + len(self.cols[end]) <= self.k:
                ins += self.cols[end]
                offs += [end - start] * len(self.cols[end])
                end += 1
            if end == start:
                raise RuntimeError("a column does not fit in one LUT")
            const = sum(self._const_bit(x) << (x - start) for x in range(start, end))
            shift = end - start if end <= self.top else self.top - start

            def fn(v, ins=ins, offs=offs, const=const, shift=shift):
                total = const + sum(
                    w * x << o for (_, _, w), x, o in zip(ins, v, offs, strict=True)
                )
                return (total >> shift) & 1

            sig = self.b.lut_lits(ins, fn)
            if end > self.top:
                return sig
            carry = (sig, False, 1)
            c = end


def compile_network(
    network: TernaryNetwork, k: int = DEFAULT_K, prune_zeros: bool = True
) -> Netlist:
    """Compile `network` into a validated, swept k-LUT netlist with identical behaviour."""
    if k < 3:
        raise ValueError("k must be at least 3 (the counters need 3-input LUTs)")
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
