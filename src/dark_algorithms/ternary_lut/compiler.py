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
  no adder result or sign bit is ever materialised. Two refinements: a step's free LUT
  inputs take bits of the next column straight into the carry when that costs no extra
  carry bit, and the carry is clamped to the values that can still change the outcome
  (anything at or above the remaining threshold, or too low to reach it with what is
  left, is folded together), which can save carry bits.
* The chain is planned as a whole rather than step by step: a small memoised search over
  the steps (how many whole columns, how many bits of the next column) also allows steps
  that take only part of the lowest column without shifting the carry, which pays off
  near the end of the chain where the clamped carry stays small. For a neuron only a few
  literals wider than a LUT this replaces the counters altogether (8 literals at k=6:
  six literals clamp to the four carry values {<=1, 2, 3, >=4}, two LUTs, then one LUT
  for the rest; the tree needed four). Each compressor tree is tried with the shallowest
  plan, the cheapest plan and the old greedy chain, so no version is worse than before.
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
  last stage, and, when it has at most 3k literals, with no counters at all.
* Counters shared across neurons: neurons of one layer that contain the same k literals
  (or all of them inverted: the count is then g - c, i.e. the counter's bits inverted
  plus a constant, which folds into the reading LUTs) read one shared k-input counter
  (full adder at   k=4) instead of building one each; groups of k-1 literals follow once
  k-literal groups run out. In a 70%-sparse layer any two neurons share about a dozen
  literals in one orientation, so most first-stage counters can be shared. Groups are
  carved as a matching: the neuron with the fewest possible partners left is served
  first, with its most constrained partner, and a group takes the common literals that
  the fewest other neurons still hold (see find_shared_groups). A neuron's
  own literals get a first stage at the same level, then the merged heap is compressed
  as above. Shared neurons must stay within a depth limit, either the depth their layer
  reaches without sharing or their own, and the network keeps the shallower (then
  smaller) of the two; a neuron with no version within its limit is built alone.
* Which version a neuron gets depends on the network's critical path: a first pass builds
  every neuron alone in its shallowest version (fewest LUTs among those) and fixes the
  network depth D; the second pass gives every neuron the cheapest version whose output
  level is at most D minus the longest path from that neuron to an output in the first
  pass (shared neurons are held to that level as well as to their sharing limit). Both
  passes run under both sharing depth policies and the shallowest, then smallest, of the
  four networks is kept.
* Identical LUTs are shared (structural hashing), constant LUTs become constant
  signals, and LUTs no output depends on are swept away.

Known headroom, roughly in order of expected payoff:
  - a still better choice of shared groups: the most-constrained-first matching shares
    83% of the literals of mlp-256-s70's first layer (pair-first greedy: 71%); an exact
    or local-search cover, groups shared across more neurons, and choosing the carving
    with the depth policies in mind (the carving moves which policy gives the shallower
    network) are open,
  - GPC shapes chosen per column by a small search instead of the greedy fill (a lazy
    schedule for every stage, not just the last, was tried and only saved 0.3%, see
    JOURNAL.md), and a last counter stage that knows which bits the planned chain will
    take (a per-column target that lets column 0 stay k bits tall saved only 0.4% per
    neuron),
  - exact synthesis for small cones,
  - required levels from a true backward timing pass over the final netlist instead of
    the first pass's paths, a shared-neuron trial cache (the benchmark now takes ~7
    minutes), and trading depth for LUTs beyond the shallowest network (mlp-256-s70 is
    now one level shallower than before at 9 more LUTs).

Setting `prune_zeros=False` compiles every weight, zero or not, as if the
hardware could not exploit sparsity. It is the baseline the eval compares against.
"""

from __future__ import annotations

import itertools
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
    def __init__(self, k: int, num_inputs: int, plans: dict | None = None) -> None:
        self.netlist = Netlist(k=k, num_inputs=num_inputs)
        self._cache: dict[tuple[tuple[int, ...], int], int] = {}
        self._level = [0] * self.netlist.first_lut_signal  # logic depth of every signal
        self._added: list[tuple[tuple[int, ...], int]] = []  # cache keys in creation order
        self._plans: dict = plans if plans is not None else {}  # comparator planners by heap

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
        self._undo(mark)
        return cost

    def trial_chains(
        self, prepare: Callable[[], BitHeap], threshold: int
    ) -> list[tuple[int, int, str]]:
        """Build a heap with `prepare` once and try every comparator chain on it.

        Returns (new LUTs, output level, plan) for the shallowest and the cheapest planned
        chain and the greedy one (see heap_at_least), then undoes everything; the compressor
        tree is built once for all three. Keeping the greedy chain means no neuron version
        is ever worse than it was before chains were planned.
        """
        mark = len(self.netlist.luts)
        heap = prepare()
        middle = len(self.netlist.luts)
        out = []
        for plan in ("shallow", "cheap", "greedy"):
            copy = BitHeap(cols={c: list(v) for c, v in heap.cols.items()}, offset=heap.offset)
            sig = self.heap_at_least(copy, threshold, plan=plan)
            out.append((len(self.netlist.luts) - mark, self._level[sig], plan))
            self._undo(middle)
        self._undo(mark)
        return out

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

    @staticmethod
    def _carry_range(
        q_lo: int, q_hi: int, v_hi: int, t_rest: int, delta: int, r_hi: int
    ) -> tuple[int, int]:
        """Range [lo, hi] of the next carry, clamped to the values that can change the output.

        The next carry is q' = floor((q + v - t_low) / 2^delta) for the current carry q and the
        step's bits v. The output is q' + rest >= t' with rest in [0, r_hi], so any q' >= t' is
        as good as t' and any q' < t' - r_hi as good as t' - r_hi - 1: clamping to that range
        never changes the output and can save carry bits (one LUT each).
        """
        t_low = t_rest & ((1 << delta) - 1)
        t_next = t_rest >> delta
        lo = max((q_lo - t_low) >> delta, t_next - r_hi - 1)
        hi = min((q_hi + v_hi - t_low) >> delta, t_next)
        return lo, hi

    def _absorb_count(
        self,
        q_lo: int,
        q_hi: int,
        v_hi: int,
        delta: int,
        pool: Sequence[Elem],
        d: int,
        t_rest: int,
        r_hi: int,
        free: int,
    ) -> int:
        """How many bits of the next column (relative weight 2^d) a comparator step absorbs.

        `delta`, `v_hi` and `r_hi` (the bits left in the heap, relative to 2^delta) describe
        the step without absorbing. As many bits are absorbed as fit into the free LUT inputs
        while the step needs no more carry bits than without them, so they leave the heap
        for free. Absorbing more at the price of a carry bit was measured as worse.
        """
        lo, hi = self._carry_range(q_lo, q_hi, v_hi, t_rest, delta, r_hi)
        base = (hi - lo).bit_length()
        r_d = r_hi >> (d - delta)  # the remaining bits all sit at relative weight >= 2^d
        best = 0
        for a in range(1, min(free, len(pool)) + 1):
            taken = sum(e.hi_value for e in pool[:a])
            lo, hi = self._carry_range(q_lo, q_hi, v_hi + (taken << d), t_rest, d, r_d - taken)
            if (hi - lo).bit_length() <= base:
                best = a
        return best

    def heap_at_least(
        self,
        heap: BitHeap,
        threshold: int,
        absorb: bool = True,
        clamp: bool = True,
        plan: str = "shallow",
    ) -> int:
        """Signal that is 1 iff the heap's value is >= threshold.

        Columns are consumed from the least significant one upwards. The state passed from
        one step to the next is the integer carry q = floor((low part - low threshold bits)
        / 2^c), kept as an offset-binary number (often one bit, sometimes a constant). Each
        step spends one LUT per carry bit over the carry and as many whole columns as fit
        into k inputs; the last step compares directly and yields the neuron output.

        With `clamp`, the carry is limited to the values that can still change the output
        (see _carry_range), and with `absorb`, a step's free LUT inputs take bits of the next
        column into the carry when that costs no extra carry bit (see _absorb_count). Both
        only ever remove heap bits or carry bits; turning them off gives the plain chain.

        With `plan` "shallow" (least depth, then fewest LUTs) or "cheap" (fewest LUTs, then
        least depth), and `absorb`, the steps are not chosen greedily but planned for the
        whole chain at once (see _plan_chain), which also allows steps that take only part
        of a column. The greedy chain ("greedy") is the fallback when no plan fits into
        k-input LUTs.
        """
        threshold -= heap.offset
        if plan != "greedy" and absorb:
            out = self._planned_chain(heap, threshold, clamp, cheap=plan == "cheap")
            if out is not None:
                return out
        k = self.netlist.k
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
            v_hi = sum(e.hi_value << sh for e, sh in elems)
            left = cols[len(take) :]
            r_hi = sum(e.hi_value << (col - c - delta) for col in left for e in heap.cols[col])
            if absorb and used < k:
                # Bits of the next column carry weight 2^d relative to column c, d >= delta,
                # so they can join the new carry directly: with the carry taken at position
                # d, floor((v + 2^d * b - t_low) / 2^d) just adds b to it. Free LUT inputs
                # are filled that way when it costs no carry bit (see _absorb_count).
                pool = heap.cols[left[0]]
                d = left[0] - c
                a = self._absorb_count(q_lo, q_hi, v_hi, delta, pool, d, t_rest, r_hi, k - used)
                if a:
                    taken = sum(e.hi_value for e in pool[:a])
                    r_hi = (r_hi >> (d - delta)) - taken
                    v_hi += taken << d
                    delta = d
                    elems = elems + [(e, d) for e in pool[:a]]
                    heap.cols[left[0]] = pool[a:]
                    signals = signals + [e.signal for e in pool[:a]]
                    values = q_lo + elem_values(elems, first_column=n)
                    if n:
                        values = values + combinations(n + len(elems))[:, :n] @ weights
            t_low = t_rest & ((1 << delta) - 1)
            nq_lo, nq_hi = self._carry_range(q_lo, q_hi, v_hi, t_rest, delta, r_hi)
            if not clamp:
                nq_lo = (q_lo - t_low) >> delta
                nq_hi = (q_hi + v_hi - t_low) >> delta
            next_q = np.clip((values - t_low) >> delta, nq_lo, nq_hi) - nq_lo
            new_carry: list[tuple[int, int]] = []
            new_lo = nq_lo
            for j in range((nq_hi - nq_lo).bit_length()):
                sig = self.lut_from_truth(signals, (next_q >> j) & 1 == 1)
                if sig == CONST1:
                    new_lo += 1 << j
                elif sig != CONST0:
                    new_carry.append((sig, j))
            carry, q_lo, c = new_carry, new_lo, c + delta

    def _plan_chain(
        self,
        order: tuple[int, ...],
        levels: tuple[tuple[int, ...], ...],
        his: tuple[tuple[int, ...], ...],
        threshold: int,
        clamp: bool,
    ) -> Callable[..., tuple[tuple[int, int, object], tuple[int, int, object]]]:
        """Plan a comparator chain over fixed heap bits; returns a memoised planner.

        The heap's columns `order` hold bits with the given arrival levels (ascending within
        a column) and maximum values. A step reads the carry, m whole columns and a bits of
        the column after them (the lowest-level ones). With a > 0 the carry is taken at that
        column's position d, so the a bits just add to it (as in _absorb_count); with m = 0
        this is a step that takes only part of the lowest column and does not shift the
        carry at all, which pays off when few bits are left and the clamped carry stays
        small (an 8-input threshold at k=6: six inputs give four carry values that matter,
        two LUTs, then one LUT over the carry and the last two inputs). Each step costs one
        LUT per carry bit and the last step one LUT.

        planner(c, q_lo, q_hi, carry_level, pos, consumed) gives two plans for the chain
        from that state as (output level, LUTs, move): the shallowest (then fewest LUTs) and
        the cheapest (then least depth). A move is (m, a), 0 or 1 for a constant output, or
        None when no chain fits into k inputs. `pos` and `consumed` say how far the columns
        are used up: columns before index pos entirely, column pos up to bit `consumed`.
        Planners are cached by heap, since trials and passes meet the same heaps again.
        """
        key = (order, levels, his, threshold, clamp)
        if key in self._plans:
            return self._plans[key]
        k = self.netlist.k
        ncol = len(order)
        none = (1 << 30, 1 << 30, None)
        # prefix[i][s]: value of column i's first s bits; above[i]: absolute value of every
        # bit in columns i.., so the value left from (pos, consumed) is exact at any column.
        prefix = [list(itertools.accumulate(h, initial=0)) for h in his]
        above = [0] * (ncol + 1)
        for i in range(ncol - 1, -1, -1):
            above[i] = above[i + 1] + (prefix[i][-1] << order[i])

        def rest_value(pos: int, consumed: int, base: int) -> int:
            if pos >= ncol:
                return 0
            return (above[pos] - (prefix[pos][consumed] << order[pos])) >> base

        @cache
        def go(c: int, q_lo: int, q_hi: int, qlev: int, pos: int, consumed: int):
            t_rest = threshold >> c
            if q_lo >= t_rest:
                return (0, 0, 1), (0, 0, 1)
            if rest_value(pos, consumed, c) + q_hi < t_rest:
                return (0, 0, 0), (0, 0, 0)
            nb = (q_hi - q_lo).bit_length()
            if pos >= ncol:
                final = (1 + qlev, 1, (0, 0)) if nb <= k else none
                return final, final
            shallow = cheap = none
            used, v, lev = nb, 0, qlev if nb else 0
            for m in range(ncol - pos + 1):
                if m:
                    i = pos + m - 1
                    s = consumed if m == 1 else 0
                    used += len(levels[i]) - s
                    if used > k:
                        break
                    v += (prefix[i][-1] - prefix[i][s]) << (order[i] - c)
                    lev = max(lev, levels[i][-1])
                    if i == ncol - 1:
                        final = (1 + lev, 1, (m, 0))
                        shallow = min(shallow, final, key=lambda x: (x[0], x[1]))
                        cheap = min(cheap, final, key=lambda x: (x[1], x[0]))
                        break
                j = pos + m
                s = consumed if m == 0 else 0
                d = order[j] - c
                for a in range(0 if m else 1, min(k - used, len(levels[j]) - s) + 1):
                    if a:
                        delta = d
                        va = v + ((prefix[j][s + a] - prefix[j][s]) << d)
                        la = max(lev, levels[j][s + a - 1])
                        npos, ncons = (j, s + a) if s + a < len(levels[j]) else (j + 1, 0)
                    else:
                        delta = order[j - 1] + 1 - c
                        va, la, npos, ncons = v, lev, j, s
                    r = rest_value(npos, ncons, c + delta)
                    if clamp:
                        lo, hi = self._carry_range(q_lo, q_hi, va, t_rest, delta, r)
                    else:
                        t_low = t_rest & ((1 << delta) - 1)
                        lo, hi = (q_lo - t_low) >> delta, (q_hi + va - t_low) >> delta
                    cost = (hi - lo).bit_length()
                    sub_shallow, sub_cheap = go(
                        c + delta, lo, hi, 1 + la if cost else 0, npos, ncons
                    )
                    for sub, best, rank in (
                        (sub_shallow, "shallow", lambda x: (x[0], x[1])),
                        (sub_cheap, "cheap", lambda x: (x[1], x[0])),
                    ):
                        if sub[2] is None:
                            continue
                        cand = (sub[0], cost + sub[1], (m, a)) if sub[0] else (0, 0, (m, a))
                        if best == "shallow":
                            shallow = min(shallow, cand, key=rank)
                        else:
                            cheap = min(cheap, cand, key=rank)
            return shallow, cheap

        self._plans[key] = go
        return go

    def _planned_chain(
        self, heap: BitHeap, threshold: int, clamp: bool, cheap: bool = False
    ) -> int | None:
        """heap_at_least along a planned chain (see _plan_chain); None if no plan fits.

        `threshold` already has the heap's offset subtracted.
        """
        order = [col for col in sorted(heap.cols) if heap.cols[col]]
        bits = [sorted(heap.cols[col], key=lambda e: self._level[e.signal]) for col in order]
        levels = tuple(tuple(self._level[e.signal] for e in b) for b in bits)
        his = tuple(tuple(e.hi_value for e in b) for b in bits)
        planner = self._plan_chain(tuple(order), levels, his, threshold, clamp)
        which = 1 if cheap else 0
        c, q_lo, q_hi, qlev, pos, consumed = 0, 0, 0, 0, 0, 0
        carry: list[tuple[int, int]] = []  # (signal, bit position) of q - q_lo
        if planner(c, q_lo, q_hi, qlev, pos, consumed)[which][2] is None:
            return None
        while True:
            move = planner(c, q_lo, q_hi, qlev, pos, consumed)[which][2]
            if move is None:  # cannot happen: real carries are never wider than planned
                raise AssertionError("comparator plan lost its way")
            if move in (0, 1):
                return CONST1 if move else CONST0
            m, a = move
            t_rest = threshold >> c
            segs = [(i, consumed if i == pos else 0) for i in range(pos, len(order))]
            elems = [(e, order[i] - c) for i, s in segs[:m] for e in bits[i][s:]]
            if a:
                j, s = segs[m]
                d = order[j] - c
                elems += [(e, d) for e in bits[j][s : s + a]]
            n = len(carry)
            signals = [sig for sig, _ in carry] + [e.signal for e, _ in elems]
            values = q_lo + elem_values(elems, first_column=n)
            if n:
                weights = np.array([1 << j for _, j in carry], dtype=np.int64)
                values = values + combinations(n + len(elems))[:, :n] @ weights
            if m == len(segs):
                return self.lut_from_truth(signals, values >= t_rest)
            if a:
                delta = d
                pos, consumed = (j, s + a) if s + a < len(bits[j]) else (j + 1, 0)
            else:
                delta = order[segs[m - 1][0]] + 1 - c
                pos, consumed = segs[m][0], 0
            v_hi = sum(e.hi_value << sh for e, sh in elems)
            r = sum(
                sum(his[i][consumed if i == pos else 0 :]) << (order[i] - c - delta)
                for i in range(pos, len(order))
            )
            t_low = t_rest & ((1 << delta) - 1)
            if clamp:
                nq_lo, nq_hi = self._carry_range(q_lo, q_hi, v_hi, t_rest, delta, r)
            else:
                nq_lo, nq_hi = (q_lo - t_low) >> delta, (q_hi + v_hi - t_low) >> delta
            next_q = np.clip((values - t_low) >> delta, nq_lo, nq_hi) - nq_lo
            new_carry: list[tuple[int, int]] = []
            new_lo = nq_lo
            for j in range((nq_hi - nq_lo).bit_length()):
                sig = self.lut_from_truth(signals, (next_q >> j) & 1 == 1)
                if sig == CONST1:
                    new_lo += 1 << j
                elif sig != CONST0:
                    new_carry.append((sig, j))
            step_level = 1 + max((self._level[sig] for sig in signals), default=0)
            carry, c = new_carry, c + delta
            q_lo = new_lo
            q_hi = min(nq_hi, new_lo + sum(1 << j for _, j in new_carry))
            qlev = step_level if new_carry else 0

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
        levels, so later passes over the network do not repeat them.
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

        def prepare(max_height: int, gpc: bool, group_size: int, lazy: bool) -> BitHeap:
            heap = BitHeap(cols={0: list(elems)})
            if max_height:
                self.compress(heap, max_height, gpc=gpc, group_size=group_size, lazy=lazy)
            return heap

        def build(max_height: int, gpc: bool, group_size: int, lazy: bool, plan: str) -> int:
            heap = prepare(max_height, gpc, group_size, lazy)
            return self.heap_at_least(heap, literal_threshold, plan=plan)

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
        if len(elems) <= 3 * k:
            # No counters at all: the planned comparator chain takes the literals directly
            # (max_height 0), which wins for neurons only a few literals wider than a LUT.
            options.append((0, False, 0, False))
        key = (tuple(terms), threshold, tuple(self._level[s] for s, _ in terms))
        tried = costs.get(key) if costs is not None else None
        if tried is None:
            tried = [
                (luts, level, (*o, plan))
                for o in options
                for luts, level, plan in self.trial_chains(
                    lambda o=o: prepare(*o), literal_threshold
                )
            ]
            if costs is not None:
                costs[key] = tried
        by_depth = min(tried, key=lambda t: (t[1], t[0], t[2]))
        if shallowest:
            return build(*by_depth[2])
        fits = [t for t in tried if max_level is None or t[1] <= max_level]
        best = min(fits, key=lambda t: (t[0], t[1], t[2])) if fits else by_depth
        return build(*best[2])

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

        def prepare(
            max_height: int, gpc: bool, group_size: int, first_stage: bool, lazy: bool
        ) -> BitHeap:
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
            if max_height:
                self.compress(heap, max_height, gpc=gpc, group_size=group_size, lazy=lazy)
            return heap

        def build(
            max_height: int,
            gpc: bool,
            group_size: int,
            first_stage: bool,
            lazy: bool,
            plan: str,
        ) -> int:
            heap = prepare(max_height, gpc, group_size, first_stage, lazy)
            return self.heap_at_least(heap, literal_threshold, plan=plan)

        outs = k.bit_length()
        group_sizes = (0, 3) if 2 * (k - outs) < outs else (0,)
        gpcs = (False, True) if k >= 5 else (False,)
        options = [
            (h, gpc, g, first, lazy)
            for lazy in (False, True)
            for first in (True, False)
            for g in group_sizes
            for gpc in gpcs
            for h in range(2, k + 1)
        ]
        bits = len(elems) + sum(len(v) for c, _, _ in shared for v in c.cols.values())
        if bits <= 3 * k:
            options.append((0, False, 0, False, False))
        limit = depth_limit if depth_limit is not None else 1 << 30
        costs = [
            (luts, level, (*o, plan))
            for o in options
            for luts, level, plan in self.trial_chains(lambda o=o: prepare(*o), literal_threshold)
        ]
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
    trees) take part. Two neurons are partners while they still have `size` common free
    literals in one orientation. Carving is a matching problem: every group uses up
    literals that other pairs could have shared, so each round serves the most constrained
    neuron first (the one with the fewest partners left, then the most free literals) and
    pairs it with its most constrained partner (then the one it has most in common with).
    Taking the pair with the most common literals instead, as an earlier version did, used
    up the literals of well-connected neurons early and left about a quarter of the
    literals of a 70%-sparse layer unshared.

    The pair's common set is narrowed while further neurons can join it with `size`
    literals left. Of the common literals, the group takes the `size` that the fewest other
    neurons still have, keeping widely held literals free for later pairings. A literal
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
        common = np.maximum(agree, disagree)
        partners = (common >= size).sum(axis=1)
        if not partners.any():
            return groups
        free_lits = mag.sum(axis=1)
        a = min(np.flatnonzero(partners), key=lambda i: (partners[i], -free_lits[i], i))
        b = min(np.flatnonzero(common[a] >= size), key=lambda j: (partners[j], -common[a, j], j))
        a, b = int(a), int(b)
        ref = free[a].copy()
        members = {a: 1, b: 1 if agree[a, b] >= disagree[a, b] else -1}
        cols = np.flatnonzero((ref != 0) & (free[b] == members[b] * ref))
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
        others = np.ones(len(free), dtype=bool)
        others[list(members)] = False
        held = mag[others][:, cols].sum(axis=0)
        cols = cols[np.argsort(held, kind="stable")][:size]
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
    required: Sequence[int | None] | None = None,
    shallowest: bool = False,
    costs: dict | None = None,
) -> list[int]:
    """Output signals of one layer, optionally with counters shared across its neurons.

    `depth_limits` caps the depth of each neuron that reads shared counters (see
    _Builder.shared_neuron); compile_network passes each neuron's depth without sharing.
    `required` gives each neuron the level its output has to reach (see compile_network),
    `shallowest` builds every neuron alone in its shallowest version, and `costs` caches
    the trial results of neurons built alone.
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
        need = required[j] if required is not None else None
        if j in per_neuron:
            # Popcount threshold: every -1 weight in a shared group is an inverted literal.
            literal_threshold = threshold + sum(1 for i in grouped if row[i] < 0)
            limits = [x for x in (depth_limits[j] if depth_limits else None, need) if x]
            limit = min(limits) if limits else None
            out = b.shared_neuron(terms, literal_threshold, per_neuron[j], limit)
        if out is None:  # not in a group, or no version around its groups is shallow enough
            all_terms = [(current[i], int(row[i])) for i in range(len(row))]
            out = b.neuron(
                [t for t in all_terms if t[1] != 0 or not prune_zeros],
                threshold,
                max_level=need,
                shallowest=shallowest,
                costs=costs,
            )
        outs.append(out)
    return outs


def _paths_to_outputs(netlist: Netlist) -> list[int]:
    """For every signal, the most LUTs on a path from it to an output (0 for an output)."""
    down = [0] * netlist.num_signals
    for n in range(len(netlist.luts) - 1, -1, -1):
        sig = netlist.first_lut_signal + n
        for s in netlist.luts[n].inputs:
            down[s] = max(down[s], 1 + down[sig])
    return down


def _split_layers(network: TernaryNetwork, signals: list[int]) -> list[list[int | None]]:
    """Cut the flat list of neuron output signals into one list per layer."""
    out, start = [], 0
    for layer in network.layers:
        out.append(list(signals[start : start + layer.out_features]))
        start += layer.out_features
    return out


def compile_network(
    network: TernaryNetwork,
    k: int = DEFAULT_K,
    prune_zeros: bool = True,
    share_counters: bool = True,
) -> Netlist:
    """Compile `network` into a validated, swept k-LUT netlist with identical behaviour.

    Counters shared across a layer's neurons save LUTs, but a neuron's cheapest version
    around them can be deeper. Two depth policies for shared neurons are compiled: each may
    reach the depth its layer reaches without sharing, or only its own depth without
    sharing. `share_counters=False` builds every neuron alone.

    For each policy the network is built twice. The first pass builds every neuron alone in
    its shallowest version, which fixes the network depth D and, for every neuron, the
    longest path from it to an output. The second pass gives every neuron the cheapest
    version that still lets its output arrive by D minus that path, so neurons off the
    critical path can use deeper, cheaper versions. Of all builds the shallowest network,
    then the smallest, is kept.
    """
    if k < 3:
        raise ValueError("k must be at least 3 (the counters need 3-input LUTs)")
    costs: dict = {}
    plans: dict = {}  # comparator planners, shared by every build of this network
    policies = (False, True) if share_counters else (None,)
    best = None
    for per_neuron in policies:
        share = per_neuron is not None
        first, signals = _compile(
            network, k, prune_zeros, bool(per_neuron), share, costs, plans=plans
        )
        down = _paths_to_outputs(first.netlist)
        depth = max((first.level(s) for s in first.netlist.outputs), default=0)
        required = [depth - down[s] for s in signals]
        second, second_signals = _compile(
            network, k, prune_zeros, bool(per_neuron), share, costs, required, plans=plans
        )
        for b, sigs in ((first, signals), (second, second_signals)):
            b.netlist.layer_outputs = _split_layers(network, sigs)
            netlist = b.netlist.sweep()
            key = (netlist.depth(), netlist.lut_count())
            if best is None or key < best[0]:
                best = (key, netlist)
    netlist = best[1]
    netlist.validate()
    return netlist


def _compile(
    network: TernaryNetwork,
    k: int,
    prune_zeros: bool,
    per_neuron: bool,
    share: bool,
    costs: dict,
    required: Sequence[int] | None = None,
    plans: dict | None = None,
) -> tuple[_Builder, list[int]]:
    """Build the network layer by layer; returns the builder and every neuron's output signal.

    With `required` None, neurons built alone get their shallowest version; otherwise neuron
    number n (counting through all layers) gets its cheapest version within level
    required[n], and shared neurons are held to that level too.
    """
    b = _Builder(k, network.num_inputs, plans)
    current = [b.netlist.input_signal(i) for i in range(network.num_inputs)]
    signals: list[int] = []
    for layer in network.layers:
        w, t = layer.weights, layer.thresholds
        need = required[len(signals) : len(signals) + len(t)] if required is not None else None

        def build(share: bool, limits: list[int] | None = None, w=w, t=t, cur=current, need=need):
            return compile_layer(
                b, w, t, cur, share, prune_zeros, limits, need, need is None, costs
            )

        if not share:
            current = build(False)
        else:
            # Sharing is kept only for layers where it pays (fewest LUTs, then least depth).
            plain_luts, levels = b.trial_levels(lambda: build(False))
            plain = (plain_luts, max(levels, default=0))
            if not per_neuron:
                levels = [plain[1]] * len(levels)
            shared = b.trial(lambda levels=levels: build(True, levels))
            current = build(True, levels) if shared < plain else build(False)
        signals.extend(current)
    b.netlist.outputs = current
    return b, signals
