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
* Counter shapes: a column's short last group (fewer than k bits removes few bits per
  LUT) can be filled with bits of the next column, a two-column counter such as (1,5;3);
  and when a k-bit counter removes less than one bit per two LUTs (k=4), columns can be
  cut into full adders (3-bit groups) instead.
* Lazy last stage: a Wallace stage compresses every column that is too tall as far as it
  can, so the last stage often leaves columns well below the target height. Those bits
  would have been cheaper in the comparator (about two bits per LUT, against one bit per
  LUT in a counter). As an option, the last stage only removes as many bits from each
  column as it needs to reach the target height, counting the bits lower columns' counters
  pass up; it is used only when it reaches the target in that one stage, so the number of
  stages is the same as Wallace's.
* Tall columns mean fewer counters but a longer chain, so every neuron is built for each
  target column height 2..k and each applicable counter shape, with and without the lazy
  last stage.
* Which version a neuron gets depends on the network's critical path: a first pass builds
  every neuron in its shallowest version (fewest LUTs among those) and fixes the network
  depth D; the second pass gives every neuron the cheapest version whose output level is
  at most D minus the longest path from that neuron to an output in the first pass. A
  neuron off the critical path can thus use a deeper, cheaper version, and the network is
  never deeper than the all-shallowest one.
* Identical LUTs are shared (structural hashing), constant LUTs become constant
  signals, and LUTs no output depends on are swept away.

Known headroom, roughly in order of expected payoff:
  - a Dadda-style schedule for all stages, not just the last one (a plain Dadda height
    sequence, which assumes uniform columns, adds stages and was worse: see JOURNAL.md),
    and GPC shapes chosen per column by a small search instead of the greedy fill,
  - sharing literal groups (counters) across neurons that read the same inputs,
  - exact synthesis for small cones,
  - required levels from a true backward timing pass over the final netlist instead of
    the first pass's paths, and trading depth for LUTs beyond the shallowest network.

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
        key = (inputs, table)
        if key not in self._cache:
            self.netlist.luts.append(Lut(inputs, table))
            self._cache[key] = self.netlist.num_signals - 1
            self._added.append(key)
            self._level.append(1 + max(self._level[s] for s in inputs))
        return self._cache[key]

    def level(self, signal: int) -> int:
        return self._level[signal]

    def _undo(self, mark: int) -> None:
        """Remove every LUT added after the netlist had `mark` LUTs."""
        first_new = self.netlist.first_lut_signal + mark
        del self.netlist.luts[mark:]
        del self._level[first_new:]
        for key in self._added[mark:]:
            del self._cache[key]
        del self._added[mark:]

    def trial(self, build: Callable[[], int]) -> tuple[int, int]:
        """Run `build`, report (new LUTs, depth of its output), then undo every LUT it added."""
        mark = len(self.netlist.luts)
        out = build()
        cost = (len(self.netlist.luts) - mark, self._level[out])
        self._undo(mark)
        return cost

    # -- popcount threshold: compressor tree + comparator chain ------------------------------

    def counter(self, elems: Sequence[Elem], above: Sequence[Elem] = ()) -> list[tuple[int, int]]:
        """Generalised parallel counter, one LUT per output bit.

        Sums `elems` (all of one column) plus `above` (bits of the next column, weight 2) and
        returns the binary result as (signal, relative column) pairs; constant output bits come
        back as CONST0/CONST1 signals and are dealt with by the caller.
        """
        weighted = [(e, 0) for e in elems] + [(e, 1) for e in above]
        signals = [e.signal for e, _ in weighted]
        top = sum(e.hi_value << sh for e, sh in weighted)
        if top == 0:
            return []

        values = elem_values(weighted)
        return [
            (self.lut_from_truth(signals, (values >> j) & 1 == 1), j)
            for j in range(top.bit_length())
        ]

    def borrow_count(self, group: Sequence[Elem], above: Sequence[Elem]) -> int:
        """How many next-column bits to add to a short counter group (a two-column GPC).

        A short group is wasteful: 3, 4 or 5 bits need 2, 3 or 3 LUTs to remove 1, 1 or 2 bits.
        Filling the free LUT inputs with bits of the next column, e.g. a (1,5;3) counter
        (max 7, three LUTs, removes three bits) or a (1,4;3) counter, removes more bits for the
        same LUTs. Picks the shape with the most bits removed per LUT, then the most removed.
        """
        k = self.netlist.k
        best_key, best = (0.0, 0), 0
        base = sum(e.hi_value for e in group)
        for a in range(min(len(above), k - len(group)) + 1):
            outs = (base + 2 * sum(e.hi_value for e in above[:a])).bit_length()
            removed = len(group) + a - outs
            if outs == 0 or removed <= 0:
                continue
            key = (removed / outs, removed)
            if key > best_key:
                best_key, best = key, a
        return best

    def compress(
        self,
        heap: BitHeap,
        max_height: int,
        gpc: bool = False,
        group_size: int = 0,
        lazy: bool = False,
    ) -> None:
        """Wallace-style reduction with k-input counters until no column exceeds max_height.

        Every stage adds one LUT level. Within a stage each column is cut into groups of up
        to k bits; a group of g bits becomes ceil(log2(g + 1)) bits spread over the next
        columns, so a k=6 group (a 6:3 counter) removes three bits for three LUTs. With
        `gpc`, a column's short last group also takes bits of the next column (see
        borrow_count). With `lazy`, a stage that can reach max_height by removing only the
        bits each column has in excess does just that (see _stage), and otherwise falls back
        to a full Wallace stage, so the stage count does not change.
        """
        while heap.max_height() > max_height:
            if lazy:
                mark = len(self.netlist.luts)
                saved = ({c: list(v) for c, v in heap.cols.items()}, heap.offset)
                self._stage(heap, max_height, gpc, group_size, lazy=True)
                if heap.max_height() <= max_height:
                    return
                self._undo(mark)
                heap.cols, heap.offset = saved
            self._stage(heap, max_height, gpc, group_size, lazy=False)

    def _stage(
        self, heap: BitHeap, max_height: int, gpc: bool, group_size: int, lazy: bool
    ) -> None:
        """One level of counters.

        Wallace (lazy=False): every column taller than max_height is cut into groups of up to
        k bits. Lazy: columns are visited from the least significant one, and a column only
        gets counters until its bits plus the counter outputs already passed up from lower
        columns fit into max_height; a group of g bits leaves one bit in its column, so it
        removes g - 1 there and is no larger than needed.
        """
        k = group_size or self.netlist.k
        # Groups of 1-2 bits remove nothing; only a target height of 1 needs half adders.
        min_group = 3 if max_height >= 2 else 2
        nxt = BitHeap(offset=heap.offset)
        for col in sorted(heap.cols):
            elems = heap.cols[col]
            if lazy:
                excess = len(elems) + len(nxt.cols.get(col, [])) - max_height
            else:
                excess = len(elems) if len(elems) > max_height else 0
            i = 0
            while excess > 0 and i < len(elems):
                size = min(k, len(elems) - i, excess + 1) if lazy else k
                group = elems[i : i + size]
                if len(group) < min_group:
                    break
                i += len(group)
                excess -= len(group) - 1 if lazy else len(group)
                above: list[Elem] = []
                if gpc and len(group) < self.netlist.k:
                    pool = heap.cols.get(col + 1, [])
                    a = self.borrow_count(group, pool)
                    above, heap.cols[col + 1] = pool[:a], pool[a:]
                for sig, j in self.counter(group, above):
                    nxt.add_signal(col + j, sig)
            nxt.extend(col, elems[i:])
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

    def neuron(
        self,
        terms: Sequence[tuple[int, int]],
        threshold: int,
        max_level: int | None = None,
        shallowest: bool = False,
        costs: dict | None = None,
    ) -> int:
        """Signal that is 1 iff sum(w * x) >= threshold.

        With x' = 1 - x for every -1 weight, sum(w * x) = popcount(literals) - #negative, so
        the neuron is a popcount threshold over literals whose inversions cost nothing.

        A wide neuron is built in the version with the fewest new LUTs (then least depth)
        whose output level is at most max_level, or, with `shallowest`, in the version with
        the least depth (then fewest LUTs). If no version meets max_level, the shallowest one
        is used. `costs` caches the trial results of a neuron whose inputs arrive at the same
        levels, so a second pass over the network does not repeat them.
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

        def build(max_height: int, gpc: bool, group_size: int, lazy: bool) -> int:
            heap = BitHeap(cols={0: list(elems)})
            self.compress(heap, max_height, gpc=gpc, group_size=group_size, lazy=lazy)
            return self.heap_at_least(heap, literal_threshold)

        # Tall columns mean fewer counter LUTs but a longer comparator chain, and the counter
        # shape that pays off depends on the fan-in, so try the combinations and keep the best.
        # Full adders (3-bit groups) only beat a k-bit counter when the latter removes less
        # than one bit per two LUTs (k=4: a (4;3) removes one bit for three LUTs), and a short
        # group can only profit from next-column bits when k >= 5 (see borrow_count).
        outs = k.bit_length()
        group_sizes = (0, 3) if 2 * (k - outs) < outs else (0,)
        gpcs = (False, True) if k >= 5 else (False,)
        options = [
            (h, gpc, g, lazy)
            for lazy in (False, True)
            for g in group_sizes
            for gpc in gpcs
            for h in range(2, k + 1)
        ]
        key = (tuple(terms), threshold, tuple(self._level[s] for s, _ in terms))
        tried = costs.get(key) if costs is not None else None
        if tried is None:
            tried = [(*self.trial(lambda o=o: build(*o)), o) for o in options]
            if costs is not None:
                costs[key] = tried
        by_depth = min(tried, key=lambda t: (t[1], t[0], t[2]))
        if shallowest:
            return build(*by_depth[2])
        fits = [t for t in tried if max_level is None or t[1] <= max_level]
        best = min(fits, key=lambda t: (t[0], t[1], t[2])) if fits else by_depth
        return build(*best[2])


def _paths_to_outputs(netlist: Netlist) -> list[int]:
    """For every signal, the most LUTs on a path from it to an output (0 for an output)."""
    down = [0] * netlist.num_signals
    for n in range(len(netlist.luts) - 1, -1, -1):
        sig = netlist.first_lut_signal + n
        for s in netlist.luts[n].inputs:
            down[s] = max(down[s], 1 + down[sig])
    return down


def _build_network(
    network: TernaryNetwork,
    k: int,
    prune_zeros: bool,
    costs: dict,
    required: Callable[[int], int | None] | None,
) -> tuple[_Builder, list[int]]:
    """Build every neuron, layer by layer; returns the builder and the neuron output signals.

    With `required` None every neuron gets its shallowest version; otherwise neuron number n
    (counting through all layers) gets its cheapest version within level required(n).
    """
    b = _Builder(k, network.num_inputs)
    current = [b.netlist.input_signal(i) for i in range(network.num_inputs)]
    neuron_signals: list[int] = []
    for layer in network.layers:
        nxt = []
        for j in range(layer.out_features):
            row = layer.weights[j]
            terms = [
                (current[i], int(row[i]))
                for i in range(layer.in_features)
                if row[i] != 0 or not prune_zeros
            ]
            n = len(neuron_signals) + len(nxt)
            limit = required(n) if required is not None else None
            nxt.append(
                b.neuron(
                    terms,
                    int(layer.thresholds[j]),
                    max_level=limit,
                    shallowest=required is None,
                    costs=costs,
                )
            )
        neuron_signals.extend(nxt)
        current = nxt
    b.netlist.outputs = current
    return b, neuron_signals


def compile_network(
    network: TernaryNetwork, k: int = DEFAULT_K, prune_zeros: bool = True
) -> Netlist:
    """Compile `network` into a validated, swept k-LUT netlist with identical behaviour.

    Two passes: the first builds every neuron in its shallowest version and so fixes the
    network depth D and, for every neuron, the longest path from it to an output. The second
    gives every neuron the cheapest version that still lets its output arrive by D minus that
    path. The first pass's version always meets this level (its paths do not depend on when
    the inputs arrive), so the result is never deeper than the first pass.
    """
    if k < 3:
        raise ValueError("k must be at least 3 (the counters need 3-input LUTs)")
    costs: dict = {}
    first, first_signals = _build_network(network, k, prune_zeros, costs, None)
    down = _paths_to_outputs(first.netlist)
    depth = max((first.level(s) for s in first.netlist.outputs), default=0)

    def required(n: int) -> int:
        return depth - down[first_signals[n]]

    b, _ = _build_network(network, k, prune_zeros, costs, required)
    netlist = b.netlist.sweep()
    netlist.validate()
    return netlist
