"""Compile a ternary network into a k-LUT netlist with no weight memory.

Strategy:

* Zero weights are dropped: a neuron only wires up its non-zero inputs. Inputs that are
  compile-time constants (an upstream neuron folded to 0 or 1) are folded into the
  threshold, and a signal feeding a neuron twice is merged into one net weight.
* A neuron whose remaining fan-in fits in one LUT (<= k inputs) becomes a single
  LUT whose truth table is the threshold function itself.
* A wider neuron is rewritten as a popcount of literals: a -1 weight on x becomes
  the literal NOT x (inversions are free, every LUT absorbs them) and the threshold
  shifts by the number of negative weights. Adding the constant 2^W - T' moves the
  threshold T' onto a power of two, so the neuron is just bit W of
  popcount + 2^W - T', and only that sum modulo 2^(W+1) is ever needed: columns
  above W are never built.
* The popcount is a bit heap (columns of equally weighted bits) compressed in stages
  by generalized parallel counters (GPCs), e.g. (6;3) and (1,5;3) for k=6, each output
  bit one LUT, until every column below W holds at most two bits. A carry chain then
  covers as many whole columns per LUT as fit in k inputs, and its last LUT produces
  bit W directly, so there is no separate sign test or inverter.
* Identical LUTs are shared (structural hashing), constant LUTs become constant
  signals, and LUTs no output depends on are swept away.

Known headroom, roughly in order of expected payoff:
  - delay-aware compression (feed late-arriving bits into the last stages),
  - grouping inputs into counters by shared structure across neurons, so counters
    over the same literals are built once,
  - exact synthesis for small neurons and for the final carry chain,
  - choosing W and the offset per neuron to minimise the chain rather than W.

Setting `prune_zeros=False` compiles every weight, zero or not, as if the
hardware could not exploit sparsity: a zero weight still occupies a counter input.
It is the baseline the eval compares against.
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
        if len(inputs) == 1 and table == 0b10:
            return inputs[0]  # a buffer: wire the input through
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
        """Signal that is 1 iff sum(w * x) >= threshold."""
        lits, target = _to_popcount(terms, threshold)
        # Now: fire iff sum(weight * literal) >= target, all weights in {0, 1}.
        hi = sum(w for _, _, w in lits)
        if target <= 0:
            return CONST1
        if target > hi:
            return CONST0
        if len(lits) <= self.netlist.k:
            return self.lut_lits(
                lits,
                lambda v: int(sum(w * b for (_, _, w), b in zip(lits, v, strict=True)) >= target),
            )
        width = 0  # W: smallest with 2^W >= target and 2^W > hi - target
        while (1 << width) < target or (1 << width) <= hi - target:
            width += 1
        return _BitHeap(self, lits, offset=(1 << width) - target, top=width).bit_top()


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


class _BitHeap:
    """Bits of equal weight per column, summed modulo 2^(top+1) to produce bit `top`."""

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
