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
* Tall columns mean fewer counters but a longer chain, so every neuron is built for each
  target column height 2..k and each applicable counter shape, and the cheapest version
  (fewest new LUTs, then least depth) is kept.
* Counters shared across neurons: neurons of one layer that contain the same k literals
  (or all of them inverted: the count is then g - c, i.e. the counter's bits inverted
  plus a constant, which folds into the reading LUTs) read one shared k-input counter
  (full adder at k=4) instead of building one each; groups of k-1 literals follow once
  k-literal groups run out. In a 70%-sparse layer any two neurons share about a dozen
  literals in one orientation, so most first-stage counters can be shared. A neuron's
  own literals get a first stage at the same level, then the merged heap is compressed
  as above. Shared neurons must stay within a depth limit, either the depth their layer
  reaches without sharing or their own, and the network keeps the shallower (then
  smaller) of the two; a neuron with no version within its limit is built alone.
* Identical LUTs are shared (structural hashing), constant LUTs become constant
  signals, and LUTs no output depends on are swept away.

Known headroom, roughly in order of expected payoff:
  - a better choice of shared groups: an exact cover of each neuron's literals instead of
    the greedy pair-first carving, groups shared across more neurons, and shared second-
    stage counters (a 70%-sparse layer still builds about 0.7 LUTs per literal),
  - GPC shapes chosen per column by a small search instead of the greedy fill (a Dadda-
    style lazy schedule was tried and only saved 0.3%, see JOURNAL.md),
  - exact synthesis for small cones,
  - choosing per-neuron structures against the network's critical path instead of
    neuron-local depth or the per-neuron depth limit.

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

    def trial(self, build: Callable[[], int | Sequence[int]]) -> tuple[int, int]:
        """Run `build`, report (new LUTs, depth of its outputs), then undo every LUT it added."""
        luts, levels = self.trial_levels(build)
        return luts, max(levels, default=0)

    def trial_levels(self, build: Callable[[], int | Sequence[int]]) -> tuple[int, list[int]]:
        """Run `build`, report (new LUTs, depth of each output), then undo every LUT it added."""
        mark = len(self.netlist.luts)
        out = build()
        outs = [out] if isinstance(out, int) else list(out)
        cost = (len(self.netlist.luts) - mark, [self._level[s] for s in outs])
        first_new = self.netlist.first_lut_signal + mark
        del self.netlist.luts[mark:]
        del self._level[first_new:]
        for key in self._added[mark:]:
            del self._cache[key]
        del self._added[mark:]
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
        self, heap: BitHeap, max_height: int, gpc: bool = False, group_size: int = 0
    ) -> None:
        """Wallace-style reduction with k-input counters until no column exceeds max_height.

        Every stage adds one LUT level. Within a stage each column is cut into groups of up
        to k bits; a group of g bits becomes ceil(log2(g + 1)) bits spread over the next
        columns, so a k=6 group (a 6:3 counter) removes three bits for three LUTs. With
        `gpc`, a column's short last group also takes bits of the next column (see
        borrow_count).
        """
        k = group_size or self.netlist.k
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
                    above: list[Elem] = []
                    if gpc and len(group) < self.netlist.k:
                        pool = heap.cols.get(col + 1, [])
                        a = self.borrow_count(group, pool)
                        above, heap.cols[col + 1] = pool[:a], pool[a:]
                    for sig, j in self.counter(group, above):
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

    def shared_counter(self, group: SharedGroup, current: Sequence[int]) -> BitHeap:
        """Build a counter shared by several neurons; returns its sum as a heap (in group order).

        The counter sums the group's literals as the group defines them. A member that reads
        every literal of the group inverted (orientation -1) sees g - count instead, which is
        the counter's bits inverted plus the constant g - (2^m - 1): the inversions fold into
        the LUTs that read the bits, so both orientations share one counter.
        """
        elems = [Elem(current[i], 0, 1) if w > 0 else Elem(current[i], 1, 0) for i, w in group.lits]
        heap = BitHeap()
        for sig, j in self.counter(elems):
            heap.add_signal(j, sig)
        return heap

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

        def build(max_height: int, gpc: bool, group_size: int) -> int:
            heap = BitHeap(cols={0: list(elems)})
            self.compress(heap, max_height=max_height, gpc=gpc, group_size=group_size)
            return self.heap_at_least(heap, literal_threshold)

        # Tall columns mean fewer counter LUTs but a longer comparator chain, and the counter
        # shape that pays off depends on the fan-in, so try the combinations and keep the best.
        # Full adders (3-bit groups) only beat a k-bit counter when the latter removes less
        # than one bit per two LUTs (k=4: a (4;3) removes one bit for three LUTs), and a short
        # group can only profit from next-column bits when k >= 5 (see borrow_count).
        outs = k.bit_length()
        group_sizes = (0, 3) if 2 * (k - outs) < outs else (0,)
        gpcs = (False, True) if k >= 5 else (False,)
        options = [(h, gpc, g) for g in group_sizes for gpc in gpcs for h in range(2, k + 1)]
        best = min(options, key=lambda o: (*self.trial(lambda: build(*o)), o))
        return build(*best)

    def shared_neuron(
        self,
        terms: Sequence[tuple[int, int]],
        literal_threshold: int,
        shared: Sequence[tuple[BitHeap, int, int]],
        depth_limit: int | None = None,
    ) -> int | None:
        """Popcount threshold over the free literals in `terms` plus shared counter sums.

        `shared` holds (counter heap, orientation, group size) for literal groups whose
        counters other neurons reuse; their literals are not in `terms`, and
        `literal_threshold` already counts their inverted literals (see compile_layer).

        The free literals get their own first stage of counters, built at the same level as
        the shared counters, so mixing the two does not lengthen the path; then the merged
        heap is compressed and compared as for any wide neuron. The cheapest version within
        `depth_limit` wins; if none fits, nothing is built and None is returned.
        """
        k = self.netlist.k
        elems = [Elem(s, 1, 0) if w < 0 else Elem(s, 0, 1 if w > 0 else 0) for s, w in terms]
        literal_threshold += sum(1 for _, w in terms if w < 0)

        def build(max_height: int, gpc: bool, group_size: int, first_stage: bool) -> int:
            heap = BitHeap(cols={0: list(elems)} if elems else {})
            if first_stage and len(elems) >= 3:
                self.compress(heap, max_height=len(elems) - 1, group_size=group_size)
            for counter, orientation, size in shared:
                if orientation > 0:
                    for col, bits in counter.cols.items():
                        heap.extend(col, bits)
                    heap.offset += counter.offset
                else:
                    # size - sum = size - offset - sum(bits): each bit b of column j becomes
                    # its inverse (1 - b) minus 2^j.
                    heap.offset += size - counter.offset
                    for col, bits in counter.cols.items():
                        heap.extend(col, [Elem(e.signal, e.hi, e.lo) for e in bits])
                        heap.offset -= len(bits) << col
            self.compress(heap, max_height=max_height, gpc=gpc, group_size=group_size)
            return self.heap_at_least(heap, literal_threshold)

        outs = k.bit_length()
        group_sizes = (0, 3) if 2 * (k - outs) < outs else (0,)
        gpcs = (False, True) if k >= 5 else (False,)
        options = [
            (h, gpc, g, first)
            for first in (True, False)
            for g in group_sizes
            for gpc in gpcs
            for h in range(2, k + 1)
        ]
        limit = depth_limit if depth_limit is not None else 1 << 30
        costs = [(*self.trial(lambda o=o: build(*o)), o) for o in options]
        fitting = [c for c in costs if c[1] <= limit]
        if not fitting:
            return None
        return build(*min(fitting)[2])


@dataclass(frozen=True)
class SharedGroup:
    """Literals (input index, weight +-1) summed by one counter that several neurons read.

    members holds (neuron index, orientation): +1 if the neuron has exactly these weights,
    -1 if it has all of them negated.
    """

    lits: tuple[tuple[int, int], ...]
    members: tuple[tuple[int, int], ...]


def find_shared_groups(weights: np.ndarray, size: int, min_fanin: int) -> list[SharedGroup]:
    """Greedily carve literal groups of `size` that two or more neurons of a layer contain.

    Only neurons with more than `min_fanin` non-zero weights (the ones built as counter
    trees) take part. Each round takes the pair of neurons with the most common free
    literals in one orientation, narrows the common set while further neurons can join it
    with `size` literals left, and turns `size` of those literals into a group. A literal
    belongs to at most one group per neuron.
    """
    w = np.asarray(weights, dtype=np.int64)
    wide = np.flatnonzero(np.count_nonzero(w, axis=1) > min_fanin)
    if len(wide) < 2:
        return []
    free = w[wide].astype(np.float64)
    groups: list[SharedGroup] = []
    while True:
        mag = np.abs(free)
        dot, both = free @ free.T, mag @ mag.T
        agree, disagree = (both + dot) / 2, (both - dot) / 2
        np.fill_diagonal(agree, 0)
        np.fill_diagonal(disagree, 0)
        pick_agree = agree.max() >= disagree.max()
        common = agree if pick_agree else disagree
        a, b = np.unravel_index(int(np.argmax(common)), common.shape)
        if common[a, b] < size:
            return groups
        ref = free[a].copy()
        members = {int(a): 1, int(b): 1 if pick_agree else -1}
        cols = np.flatnonzero((ref != 0) & (free[b] == members[int(b)] * ref))
        while True:
            sub = free[:, cols]
            same = (sub == ref[cols]).sum(axis=1)
            flip = (sub == -ref[cols]).sum(axis=1)
            score = np.maximum(same, flip)
            score[list(members)] = -1
            c = int(np.argmax(score))
            if score[c] < size:
                break
            orient = 1 if same[c] >= flip[c] else -1
            members[c] = orient
            cols = cols[free[c, cols] == orient * ref[cols]]
        cols = cols[:size]
        for m, orient in members.items():
            assert (free[m, cols] == orient * ref[cols]).all()
            free[m, cols] = 0
        groups.append(
            SharedGroup(
                lits=tuple((int(i), int(ref[i])) for i in cols),
                members=tuple(sorted((int(wide[m]), o) for m, o in members.items())),
            )
        )


def shared_group_size(k: int) -> int:
    """Counter size worth sharing: k bits, or a full adder when a k-bit counter is poor (k=4)."""
    outs = k.bit_length()
    return 3 if 2 * (k - outs) < outs else k


def compile_layer(
    b: _Builder,
    weights: np.ndarray,
    thresholds: np.ndarray,
    current: list[int],
    share: bool,
    prune_zeros: bool = True,
    depth_limits: Sequence[int] | None = None,
) -> list[int]:
    """Output signals of one layer, optionally with counters shared across its neurons.

    `depth_limits` caps the depth of each neuron that reads shared counters (see
    _Builder.shared_neuron); compile_network passes each neuron's depth without sharing.
    """
    k = b.netlist.k
    groups = []
    if share:
        size = shared_group_size(k)
        groups = find_shared_groups(weights, size, k)
        if size > 5:
            used = np.zeros(weights.shape, dtype=bool)
            for g in groups:
                for m, _ in g.members:
                    used[m, [i for i, _ in g.lits]] = True
            rest = np.where(used, 0, weights)
            wide = np.count_nonzero(weights, axis=1) > k
            rest[~wide] = 0
            groups += find_shared_groups(rest, size - 1, 0)
    per_neuron: dict[int, list[tuple[BitHeap, int, int]]] = {}
    in_group: dict[int, set[int]] = {}
    for g in groups:
        counter = b.shared_counter(g, current)
        for m, orient in g.members:
            per_neuron.setdefault(m, []).append((counter, orient, len(g.lits)))
            in_group.setdefault(m, set()).update(i for i, _ in g.lits)
    outs = []
    for j in range(weights.shape[0]):
        row = weights[j]
        grouped = in_group.get(j, set())
        terms = [
            (current[i], int(row[i]))
            for i in range(len(row))
            if i not in grouped and (row[i] != 0 or not prune_zeros)
        ]
        threshold = int(thresholds[j])
        out = None
        if j in per_neuron:
            # Popcount threshold: every -1 weight in a shared group is an inverted literal.
            literal_threshold = threshold + sum(1 for i in grouped if row[i] < 0)
            limit = depth_limits[j] if depth_limits is not None else None
            out = b.shared_neuron(terms, literal_threshold, per_neuron[j], limit)
        if out is None:  # not in a group, or no version around its groups is shallow enough
            all_terms = [(current[i], int(row[i])) for i in range(len(row))]
            out = b.neuron([t for t in all_terms if t[1] != 0 or not prune_zeros], threshold)
        outs.append(out)
    return outs


def compile_network(
    network: TernaryNetwork,
    k: int = DEFAULT_K,
    prune_zeros: bool = True,
    share_counters: bool = True,
) -> Netlist:
    """Compile `network` into a validated, swept k-LUT netlist with identical behaviour.

    Counters shared across a layer's neurons save LUTs, but a neuron's cheapest version
    around them can be deeper. Two depth policies are compiled and the shallower network
    (then the smaller) is kept: each shared neuron may reach the depth its layer reaches
    without sharing, which saves the most LUTs, or only its own depth without sharing,
    which can never deepen the network. `share_counters=False` builds every neuron alone.
    """
    if k < 3:
        raise ValueError("k must be at least 3 (the counters need 3-input LUTs)")
    if not share_counters:
        return _compile(network, k, prune_zeros, share=False)
    best = None
    for per_neuron in (False, True):
        netlist = _compile(network, k, prune_zeros, per_neuron=per_neuron)
        key = (netlist.depth(), netlist.lut_count())
        if best is None or key < best[0]:
            best = (key, netlist)
    return best[1]


def _compile(
    network: TernaryNetwork, k: int, prune_zeros: bool, per_neuron: bool = True, share: bool = True
) -> Netlist:
    b = _Builder(k, network.num_inputs)
    current = [b.netlist.input_signal(i) for i in range(network.num_inputs)]
    for layer in network.layers:
        w, t = layer.weights, layer.thresholds

        def build(share: bool, limits: list[int] | None = None, w=w, t=t, cur=current):
            return compile_layer(b, w, t, cur, share, prune_zeros, limits)

        if not share:
            current = build(False)
            continue
        # Sharing is kept only for layers where it pays (fewest LUTs, then least depth).
        plain_luts, levels = b.trial_levels(lambda: build(False))
        plain = (plain_luts, max(levels, default=0))
        if not per_neuron:
            levels = [plain[1]] * len(levels)
        shared = b.trial(lambda levels=levels: build(True, levels))
        current = build(True, levels) if shared < plain else build(False)
    b.netlist.outputs = current
    netlist = b.netlist.sweep()
    netlist.validate()
    return netlist
