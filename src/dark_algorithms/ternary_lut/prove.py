"""Formal equivalence proof: does the netlist compute exactly the reference network?

Simulation can only sample a wide network's 2^n inputs. This module proves equivalence
for every input with reduced ordered binary decision diagrams (BDDs), which are canonical:
two functions are equal exactly when their BDDs are the same node.

The proof is compositional, one neuron at a time:

* For layer L, the signals carrying layer L-1's neuron outputs (the primary inputs for the
  first layer) form a cut. Every cut signal is treated as a free variable.
* For each neuron of layer L, the BDD of its output signal is built from its LUT cone,
  stopping at the cut, and compared with the BDD of the reference threshold function
  sum(w_i * x_i) >= theta over the same variables.
* If every neuron of every layer matches, the netlist equals the network on every input:
  by induction, layer L-1's cut signals carry exactly layer L-1's neuron outputs, and layer
  L's outputs are the threshold functions of those values.

Treating cut signals as free variables only over-approximates the values they can take, so
a successful proof is sound. Where the compiler folded two hidden neurons into one signal
(identical functions) or a constant, the reference uses that one variable or constant too,
which is sound because the previous layer's proof shows the neurons really are equal.

A BDD of a wide neuron's whole cone over its inputs can be huge: the middle stages of a
compressor tree compute bits of several partial sums at once (one 77-input neuron needed
two million nodes). So each neuron is proven in two steps:

1. Arithmetic reduction. The neuron's weighted sum is kept as a heap, an integer
   expression const + sum(c_s * s) over signals, starting from the reference weights on the
   cut signals. Walking the cone from its inputs, a group of LUTs that read the same inputs
   (a counter) replaces those inputs in the heap if some integers a_o and d satisfy
   sum(c_i * in_i) == sum(a_o * out_o) + d for all 2^n values of its n inputs, which is
   checked exhaustively. Each step keeps the heap equal to the weighted sum for every
   input, whatever the group's inputs are, so no knowledge of the compiler is assumed.
   A group is only replaced when nothing else in the cone reads its inputs.
2. The logic left over (the comparator) is proven with a BDD: the neuron output, as a
   function of the remaining heap signals treated as free variables, must equal
   const + sum(c_s * s) >= theta. Variables are ordered by weight, least significant first,
   so the comparator's carries keep the BDD small.

Together: output == (heap >= theta) == (weighted sum >= theta) for every input. If the
reduction finds nothing to replace, step 2 is a plain BDD of the whole cone, ordered by a
depth-first walk that keeps each counter's inputs adjacent.

Every neuron gets its own BDD manager with a node budget; a neuron whose BDDs outgrow the
budget, or a case that exceeds its time box, is reported as not proven rather than as wrong.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

from .model import TernaryNetwork
from .netlist import CONST0, CONST1, Netlist

FALSE, TRUE = 0, 1


class BudgetExceeded(Exception):
    """A BDD grew beyond its node budget or the proof ran out of time."""


class Bdd:
    """A minimal reduced ordered BDD manager without complement edges.

    Nodes are integers; 0 and 1 are the terminals. Variable 0 is tested first.
    """

    def __init__(self, num_vars: int, max_nodes: int = 400_000, deadline: float | None = None):
        self.num_vars = num_vars
        self.max_nodes = max_nodes
        self.deadline = deadline
        self._var = [num_vars, num_vars]
        self._lo = [0, 1]
        self._hi = [0, 1]
        self._unique: dict[tuple[int, int, int], int] = {}
        self._ite: dict[tuple[int, int, int], int] = {}

    @property
    def size(self) -> int:
        return len(self._var)

    def mk(self, v: int, lo: int, hi: int) -> int:
        if lo == hi:
            return lo
        key = (v, lo, hi)
        node = self._unique.get(key)
        if node is None:
            node = len(self._var)
            if node >= self.max_nodes:
                raise BudgetExceeded(f"more than {self.max_nodes} BDD nodes")
            if self.deadline is not None and node % 4096 == 0 and time.monotonic() > self.deadline:
                raise BudgetExceeded("time box exceeded")
            self._var.append(v)
            self._lo.append(lo)
            self._hi.append(hi)
            self._unique[key] = node
        return node

    def var(self, v: int) -> int:
        return self.mk(v, FALSE, TRUE)

    def ite(self, f: int, g: int, h: int) -> int:
        """If f then g else h."""
        if f == TRUE:
            return g
        if f == FALSE:
            return h
        if g == h:
            return g
        if g == TRUE and h == FALSE:
            return f
        key = (f, g, h)
        res = self._ite.get(key)
        if res is not None:
            return res
        var = self._var
        v = min(var[f], var[g], var[h])
        f0, f1 = (self._lo[f], self._hi[f]) if var[f] == v else (f, f)
        g0, g1 = (self._lo[g], self._hi[g]) if var[g] == v else (g, g)
        h0, h1 = (self._lo[h], self._hi[h]) if var[h] == v else (h, h)
        res = self.mk(v, self.ite(f0, g0, h0), self.ite(f1, g1, h1))
        self._ite[key] = res
        return res

    def lut(self, table: int, args: list[int]) -> int:
        """BDD of a LUT with the given truth table applied to the argument BDDs.

        Bit idx of `table` is the output for idx = sum(arg_j << j), as in the netlist.
        """
        memo: dict[tuple[int, int], int] = {}

        def rec(j: int, sub: int) -> int:
            if sub == 0:
                return FALSE
            if sub == (1 << (1 << j)) - 1:
                return TRUE
            key = (j, sub)
            if key not in memo:
                half = 1 << (j - 1)
                lo = sub & ((1 << half) - 1)
                memo[key] = self.ite(args[j - 1], rec(j - 1, sub >> half), rec(j - 1, lo))
            return memo[key]

        return rec(len(args), table)

    def threshold(self, weights: list[int], theta: int) -> int:
        """BDD of sum(weights[v] * x_v) >= theta, with x_v the BDD variable v."""
        n = len(weights)
        # Largest and smallest sum the variables from v onwards can still add.
        up = [0] * (n + 1)
        down = [0] * (n + 1)
        for v in range(n - 1, -1, -1):
            up[v] = up[v + 1] + max(weights[v], 0)
            down[v] = down[v + 1] + min(weights[v], 0)
        memo: dict[tuple[int, int], int] = {}

        def rec(v: int, acc: int) -> int:
            if acc + down[v] >= theta:
                return TRUE
            if acc + up[v] < theta:
                return FALSE
            key = (v, acc)
            if key not in memo:
                memo[key] = self.mk(v, rec(v + 1, acc), rec(v + 1, acc + weights[v]))
            return memo[key]

        return rec(0, 0)

    def satisfy(self, f: int) -> dict[int, int]:
        """One assignment (variable -> bit) that makes f true; f must not be FALSE."""
        out: dict[int, int] = {}
        while f > TRUE:
            v = self._var[f]
            if self._hi[f] != FALSE:
                out[v], f = 1, self._hi[f]
            else:
                out[v], f = 0, self._lo[f]
        return out

    def xor(self, f: int, g: int) -> int:
        return self.ite(f, self.ite(g, FALSE, TRUE), g)


@dataclass(frozen=True)
class ProofResult:
    """proven: every neuron's cone equals its threshold function over the layer's inputs.

    refuted: a neuron's cone differs from its reference for some assignment of its layer's
    inputs. In the first layer that is a real counterexample (`counterexample` holds the
    input vector); in later layers the differing inputs may be unreachable, so it only means
    the proof failed. `reason` says why a case is not proven.
    """

    proven: bool
    refuted: bool
    neurons_proven: int
    neurons_total: int
    seconds: float
    reason: str = ""
    counterexample: tuple[int, ...] | None = None


def _cone_leaves(netlist: Netlist, root: int, cut: set[int]) -> list[int]:
    """Leaves of root's LUT cone (cut signals and primary inputs) in depth-first order."""
    first = netlist.first_lut_signal
    seen: set[int] = set()
    leaves: list[int] = []
    stack = [root]
    while stack:
        s = stack.pop()
        if s in seen or s in (CONST0, CONST1):
            continue
        seen.add(s)
        if s in cut or s < first:
            leaves.append(s)
            continue
        # Reversed so the LUT's first input is walked first.
        stack.extend(reversed(netlist.luts[s - first].inputs))
    return leaves


def _cone_bdd(bdd: Bdd, netlist: Netlist, root: int, var_of: dict[int, int]) -> int:
    """BDD of root over the variables in var_of; the cone stops at those signals."""
    first = netlist.first_lut_signal
    memo: dict[int, int] = {CONST0: FALSE, CONST1: TRUE}
    for s, v in var_of.items():
        memo[s] = bdd.var(v)
    # Iterative post-order over the cone, so deep netlists don't hit the recursion limit.
    stack = [(root, False)]
    while stack:
        s, ready = stack.pop()
        if s in memo:
            continue
        lut = netlist.luts[s - first]
        if ready:
            memo[s] = bdd.lut(lut.table, [memo[i] for i in lut.inputs])
            continue
        stack.append((s, True))
        stack.extend((i, False) for i in lut.inputs if i not in memo)
    return memo[root]


def _cone(netlist: Netlist, root: int, stop: set[int]) -> list[int]:
    """LUT signals of root's cone, not entering `stop`, in topological (creation) order."""
    first = netlist.first_lut_signal
    seen: set[int] = set()
    stack = [root]
    while stack:
        s = stack.pop()
        if s in seen or s < first or s in stop:
            continue
        seen.add(s)
        stack.extend(netlist.luts[s - first].inputs)
    return sorted(seen)


def _solve_group(
    netlist: Netlist, outs: list[int], ins: list[int], coef: dict[int, int]
) -> tuple[dict[int, int], int] | None:
    """Integers a (per output) and d with sum(coef[i] * i) == sum(a[o] * o) + d on all inputs.

    Returns None when no such integers exist (checked exactly over all 2^len(ins) rows).
    """
    first = netlist.first_lut_signal
    rows = 1 << len(ins)
    pos = {sig: b for b, sig in enumerate(ins)}
    target = [sum(coef[sig] for b, sig in enumerate(ins) if r >> b & 1) for r in range(rows)]
    columns: list[tuple[int, list[int]]] = []
    for o in outs:
        lut = netlist.luts[o - first]
        perm = [pos[i] for i in lut.inputs]
        col = []
        for r in range(rows):
            idx = sum((r >> b & 1) << j for j, b in enumerate(perm))
            col.append(lut.table >> idx & 1)
        if any(col) and col not in [c for _, c in columns]:
            columns.append((o, col))
    matrix = np.array([[c[r] for _, c in columns] + [1] for r in range(rows)], dtype=float)
    sol = np.linalg.lstsq(matrix, np.array(target, dtype=float), rcond=None)[0]
    a = [round(x) for x in sol[:-1]]
    d = round(sol[-1])
    for r in range(rows):
        if sum(ai * c[r] for ai, (_, c) in zip(a, columns, strict=True)) + d != target[r]:
            return None
    return {o: ai for ai, (o, _) in zip(a, columns, strict=True) if ai}, d


def _group_image(netlist: Netlist, outs: list[int], ins: list[int]) -> set[tuple[int, ...]]:
    """Every value the LUTs `outs`, which read only `ins`, take together over all 2^len(ins) rows.

    Whatever values the inputs really take, the outputs always form one of these tuples.
    """
    first = netlist.first_lut_signal
    pos = {sig: b for b, sig in enumerate(ins)}
    image = set()
    for r in range(1 << len(ins)):
        row = []
        for o in outs:
            lut = netlist.luts[o - first]
            idx = sum((r >> pos[i] & 1) << j for j, i in enumerate(lut.inputs))
            row.append(lut.table >> idx & 1)
        image.add(tuple(row))
    return image


def _care_set(
    bdd: Bdd, netlist: Netlist, groups: list[tuple[list[int], list[int]]], var_of: dict[int, int]
) -> int:
    """BDD of the heap values the replaced groups can produce at all (see prove_equivalence).

    A replaced counter's outputs are BDD variables of the comparator check, but not free
    ones: a 6-input counter never outputs 7, for example, and a compiler that knows the
    counter's range may build a comparator that differs from the threshold there. Whatever
    the circuit's inputs, each group's outputs lie in its image, so the values of all groups
    together lie in the product of their images: checking the comparator only there is
    still sound.
    """
    care = TRUE
    for outs, ins in groups:
        outs = [o for o in outs if o in var_of]
        if not outs:
            continue
        group = FALSE
        for row in _group_image(netlist, outs, ins):
            term = TRUE
            for o, bit in zip(outs, row, strict=True):
                v = bdd.var(var_of[o])
                term = bdd.ite(term, v if bit else bdd.ite(v, FALSE, TRUE), FALSE)
            group = bdd.ite(group, TRUE, term)
        care = bdd.ite(care, group, FALSE)
    return care


def _reduce_heap(
    netlist: Netlist, root: int, cut: set[int], coef: dict[int, int]
) -> tuple[dict[int, int], int, list[tuple[list[int], list[int]]]]:
    """Replace counters in root's cone by their outputs while the heap value stays the same.

    Returns the new heap coefficients, the constant it gained and the replaced groups as
    (outputs, inputs).
    """
    first = netlist.first_lut_signal
    coef = dict(coef)
    const = 0
    cone = _cone(netlist, root, cut)
    readers: dict[int, set[int]] = {}
    groups: dict[frozenset[int], list[int]] = {}
    for s in cone:
        ins = netlist.luts[s - first].inputs
        groups.setdefault(frozenset(ins), []).append(s)
        for i in ins:
            readers.setdefault(i, set()).add(s)
    replaced: list[tuple[list[int], list[int]]] = []
    for key in sorted(groups, key=lambda g: min(groups[g])):
        outs = groups[key]
        ins = sorted(key)
        if root in outs or not all(coef.get(i) for i in ins) or len(ins) > 8:
            continue
        if any(not readers[i] <= set(outs) for i in ins):
            continue  # an input is also read elsewhere in the cone; keep it in the heap
        solved = _solve_group(netlist, outs, ins, coef)
        if solved is None:
            continue
        new, d = solved
        for i in ins:
            del coef[i]
        for o, a in new.items():
            coef[o] = coef.get(o, 0) + a
        const += d
        replaced.append((outs, ins))
    return {s: c for s, c in coef.items() if c}, const, replaced


def prove_equivalence(
    network: TernaryNetwork,
    netlist: Netlist,
    time_box: float = 300.0,
    max_nodes: int = 400_000,
) -> ProofResult:
    """Prove that `netlist` computes `network` on every input, layer by layer (see module doc).

    Needs `netlist.layer_outputs`, which compile_network records.
    """
    start = time.monotonic()
    deadline = start + time_box
    total = sum(layer.out_features for layer in network.layers)
    done = 0

    def result(proven: bool, reason: str = "", refuted: bool = False, cex=None) -> ProofResult:
        return ProofResult(
            proven, refuted, done, total, round(time.monotonic() - start, 2), reason, cex
        )

    if len(netlist.layer_outputs) != len(network.layers):
        return result(False, "the netlist does not record its layer outputs")
    if list(netlist.layer_outputs[-1]) != list(netlist.outputs):
        return result(False, "the last layer's outputs are not the netlist outputs")

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 10_000))
    try:
        cut_signals: list[int | None] = [netlist.input_signal(i) for i in range(network.num_inputs)]
        for li, layer in enumerate(network.layers):
            cut = {s for s in cut_signals if s is not None and s not in (CONST0, CONST1)}
            for j in range(layer.out_features):
                root = netlist.layer_outputs[li][j]
                if root is None:
                    # Swept away because no output depends on it: nothing to prove.
                    done += 1
                    continue
                if time.monotonic() > deadline:
                    return result(False, f"time box of {time_box:.0f} s exceeded")
                row = layer.weights[j]
                theta = int(layer.thresholds[j])
                # Net weight per cut signal; constant inputs move the threshold.
                weight_of: dict[int, int] = {}
                for i in np.nonzero(row)[0]:
                    s = cut_signals[i]
                    if s is None:
                        # Swept away, so no LUT reads it; it can only be irrelevant here,
                        # which the proof checks by giving it a variable of its own.
                        s = -1 - int(i)
                    if s == CONST1:
                        theta -= int(row[i])
                    elif s != CONST0:
                        weight_of[s] = weight_of.get(s, 0) + int(row[i])
                heap, const, replaced = _reduce_heap(netlist, root, cut, weight_of)
                stop = cut | set(heap)
                leaves = _cone_leaves(netlist, root, stop) if root not in stop else [root]
                if replaced:
                    # Least significant heap bits first, then anything the cone reads besides.
                    order = sorted(heap, key=lambda s: (abs(heap[s]), s))
                    order += [s for s in leaves if s not in heap]
                else:
                    order = leaves + sorted(set(heap) - set(leaves))
                var_of = {s: v for v, s in enumerate(order)}
                bdd = Bdd(len(order), max_nodes=max_nodes, deadline=deadline)
                try:
                    ref = bdd.threshold([heap.get(s, 0) for s in order], theta - const)
                    got = _cone_bdd(bdd, netlist, root, var_of)
                    care = _care_set(bdd, netlist, replaced, var_of)
                    wrong = bdd.ite(care, bdd.xor(ref, got), FALSE)
                except BudgetExceeded as exc:
                    return result(False, f"layer {li} neuron {j}: {exc}")
                if wrong != FALSE:
                    cex = None
                    if li == 0 and not replaced:
                        assign = bdd.satisfy(wrong)
                        bits = [0] * network.num_inputs
                        for s, v in var_of.items():
                            if s >= 0:
                                bits[s - netlist.input_signal(0)] = assign.get(v, 0)
                        cex = tuple(bits)
                    return result(
                        False,
                        f"layer {li} neuron {j} differs from its threshold function",
                        refuted=True,
                        cex=cex,
                    )
                done += 1
            cut_signals = list(netlist.layer_outputs[li])
    finally:
        sys.setrecursionlimit(old_limit)
    return result(True)
