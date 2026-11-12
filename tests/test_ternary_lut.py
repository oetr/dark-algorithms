import itertools

import numpy as np
import pytest

from dark_algorithms.ternary_lut import (
    Lut,
    Netlist,
    TernaryLayer,
    TernaryNetwork,
    check_equivalence,
    compile_network,
    evaluate,
    random_network,
    simulate,
)
from dark_algorithms.ternary_lut.compiler import (
    BitHeap,
    Elem,
    _Builder,
    _compile,
    find_shared_groups,
)
from dark_algorithms.ternary_lut.netlist import CONST0, CONST1


def all_vectors(n: int) -> np.ndarray:
    return np.array(list(itertools.product((0, 1), repeat=n)), dtype=np.uint8)


# -- model -------------------------------------------------------------------------------------


def test_layer_rejects_non_ternary_weights():
    with pytest.raises(ValueError, match="ternary"):
        TernaryLayer(weights=np.array([[2, 0]]), thresholds=np.array([0]))


def test_network_rejects_mismatched_layers():
    a = TernaryLayer(np.zeros((3, 4), dtype=np.int8), np.zeros(3))
    b = TernaryLayer(np.zeros((2, 5), dtype=np.int8), np.zeros(2))
    with pytest.raises(ValueError, match="mismatch"):
        TernaryNetwork((a, b))


def test_reference_forward_is_threshold_function():
    layer = TernaryLayer(np.array([[1, -1, 0]]), np.array([1]))
    x = np.array([[1, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=np.uint8)
    assert layer.forward(x).ravel().tolist() == [1, 0, 0]


def test_random_network_sparsity_and_balance():
    net = random_network([64, 64, 32], sparsity=0.8, rng=np.random.default_rng(0))
    assert 0.75 < net.sparsity() < 0.85
    out = net.forward(np.random.default_rng(1).integers(0, 2, (2000, 64), dtype=np.uint8))
    # Not collapsed to constants: the equivalence check must exercise both values.
    assert 0.2 < out.mean() < 0.8


# -- netlist -----------------------------------------------------------------------------------


def test_netlist_validate_rejects_forward_reference():
    nl = Netlist(k=2, num_inputs=1, luts=[Lut((4,), 0b01)], outputs=[3])
    with pytest.raises(ValueError, match="not defined before"):
        nl.validate()


def test_netlist_validate_rejects_too_many_inputs():
    nl = Netlist(k=2, num_inputs=3, luts=[Lut((2, 3, 4), 0b1)], outputs=[5])
    with pytest.raises(ValueError, match="more than k"):
        nl.validate()


def test_sweep_removes_dead_luts():
    nl = Netlist(
        k=2,
        num_inputs=2,
        luts=[Lut((2, 3), 0b1000), Lut((2, 3), 0b0110)],  # AND (dead), XOR (live)
        outputs=[5],
    )
    swept = nl.sweep()
    assert swept.lut_count() == 1
    assert simulate(swept, all_vectors(2)).ravel().tolist() == [0, 1, 1, 0]


def test_depth_counts_lut_levels():
    nl = Netlist(k=2, num_inputs=2, luts=[Lut((2, 3), 0b1000), Lut((4, 2), 0b0110)], outputs=[5])
    assert nl.depth() == 2


# -- compiler building blocks ------------------------------------------------------------------


def test_lut_builder_folds_constants_and_shares_identical_luts():
    b = _Builder(k=4, num_inputs=2)
    s0, s1 = b.netlist.input_signal(0), b.netlist.input_signal(1)
    assert b.lut((s0, s1), lambda v: 0) == CONST0
    assert b.lut((s0, s1), lambda v: 1) == CONST1
    first = b.lut((s0, s1), lambda v: v[0] & v[1])
    again = b.lut((s0, s1), lambda v: v[0] & v[1])
    assert first == again and b.netlist.lut_count() == 1


def test_counter_is_exact_over_all_inputs():
    b = _Builder(k=6, num_inputs=6)
    sig = [b.netlist.input_signal(i) for i in range(6)]
    lohi = [(0, 1), (1, 0), (0, 1), (1, 0), (0, 0), (0, 1)]  # +1, -1, +1, -1, 0, +1
    outs = b.counter([Elem(s, lo, hi) for s, (lo, hi) in zip(sig, lohi, strict=True)])
    assert [j for _, j in outs] == [0, 1, 2]  # 5 non-zero literals need 3 bits
    b.netlist.outputs = [s for s, _ in outs]
    x = all_vectors(6)
    value = (simulate(b.netlist, x).astype(np.int64) << np.arange(3)).sum(axis=1)
    lo = np.array([p[0] for p in lohi])
    hi = np.array([p[1] for p in lohi])
    assert (value == np.where(x == 1, hi, lo).sum(axis=1)).all()


def test_two_column_counter_is_exact_over_all_inputs():
    # A (1,5;3) counter: five bits of this column (one inverted) plus one of the next.
    b = _Builder(k=6, num_inputs=6)
    sig = [b.netlist.input_signal(i) for i in range(6)]
    here = [Elem(sig[0]), Elem(sig[1], 1, 0), Elem(sig[2]), Elem(sig[3]), Elem(sig[4])]
    outs = b.counter(here, above=[Elem(sig[5])])
    assert [j for _, j in outs] == [0, 1, 2]  # max 5 + 2 = 7 fits in three bits
    b.netlist.outputs = [s for s, _ in outs]
    x = all_vectors(6).astype(np.int64)
    value = (simulate(b.netlist, x).astype(np.int64) << np.arange(3)).sum(axis=1)
    expected = x[:, 0] + (1 - x[:, 1]) + x[:, 2] + x[:, 3] + x[:, 4] + 2 * x[:, 5]
    assert (value == expected).all()


def test_short_groups_borrow_next_column_bits_only_when_it_pays():
    b6, b4 = _Builder(k=6, num_inputs=12), _Builder(k=4, num_inputs=12)
    bits = [Elem(b6.netlist.input_signal(i)) for i in range(12)]
    assert b6.borrow_count(bits[:5], bits[5:]) == 1  # (1,5;3): removes 3 bits for 3 LUTs
    assert b6.borrow_count(bits[:4], bits[4:]) == 1  # (1,4;3): removes 2 bits for 3 LUTs
    assert b6.borrow_count(bits[:3], bits[3:]) == 2  # (2,3;3): removes 2 bits for 3 LUTs
    assert b6.borrow_count(bits[:5], []) == 0
    assert b4.borrow_count(bits[:3], bits[3:]) == 0  # a full adder is already better


@pytest.mark.parametrize("k", [4, 5, 6])
@pytest.mark.parametrize(("gpc", "group_size"), [(False, 0), (True, 0), (False, 3), (True, 3)])
@pytest.mark.parametrize(("lazy", "max_height"), [(False, 2), (True, 2), (True, 3)])
def test_compression_variants_are_exact(k, gpc, group_size, lazy, max_height):
    # 7 bits of weight 1, 6 of weight 2 (one inverted), 3 of weight 4: value range 0..31.
    n_inputs = 16
    weights = np.array([1] * 7 + [2] * 6 + [4] * 3)
    x = all_vectors(n_inputs).astype(np.int64)
    value = x @ weights + (1 - 2 * x[:, 7]) * 2
    for threshold in range(-1, 34, 3):
        b = _Builder(k=k, num_inputs=n_inputs)
        heap = BitHeap()
        for i, w in enumerate(weights):
            s = b.netlist.input_signal(i)
            heap.extend(int(w).bit_length() - 1, [Elem(s, 1, 0) if i == 7 else Elem(s)])
        b.compress(heap, max_height=max_height, gpc=gpc, group_size=group_size, lazy=lazy)
        assert heap.max_height() <= max_height
        b.netlist.outputs = [b.heap_at_least(heap, threshold)]
        out = simulate(b.netlist.sweep(), x).ravel()
        assert (out == (value >= threshold)).all(), (k, gpc, group_size, lazy, threshold)


@pytest.mark.parametrize(("n", "max_height", "wallace", "lazy"), [(9, 4, 6, 4), (16, 6, 12, 9)])
def test_lazy_last_stage_leaves_bits_to_the_comparator(n, max_height, wallace, lazy):
    # The last Wallace stage over-compresses (9 bits -> 6:3 + 3:2 counters); the lazy stage
    # only removes the excess over max_height, with the same number of stages.
    x = all_vectors(n)
    results = {}
    for mode in (False, True):
        b = _Builder(k=6, num_inputs=n)
        heap = BitHeap(cols={0: [Elem(b.netlist.input_signal(i)) for i in range(n)]})
        b.compress(heap, max_height=max_height, lazy=mode)
        out = b.heap_at_least(heap, n // 2)
        b.netlist.outputs = [out]
        assert (simulate(b.netlist, x).ravel() == (x.sum(axis=1) >= n // 2)).all()
        results[mode] = (b.netlist.lut_count(), b.level(out))
    assert results[False] == (wallace, results[True][1])
    assert results[True][0] == lazy


def test_full_adders_shrink_wide_neurons_at_k4():
    # 36-input majority at k=4: (4;3) counters needed 77 LUTs at depth 9.
    net = TernaryNetwork((TernaryLayer(np.ones((1, 36), dtype=np.int8), np.array([18])),))
    nl = compile_network(net, k=4)
    assert check_equivalence(net, nl, num_samples=8192).equivalent
    assert nl.lut_count() <= 62 and nl.depth() <= 8


@pytest.mark.parametrize("k", [3, 4, 6])
def test_heap_comparator_is_exact_for_every_threshold(k):
    # 5 bits of weight 1, 4 of weight 2 and 3 of weight 4: value range 0..25.
    n_inputs = 12
    weights = np.array([1] * 5 + [2] * 4 + [4] * 3)
    x = all_vectors(n_inputs)
    value = x @ weights
    for threshold in range(-1, 28):
        b = _Builder(k=k, num_inputs=n_inputs)
        sig = [b.netlist.input_signal(i) for i in range(n_inputs)]
        heap = BitHeap()
        for s, w in zip(sig, weights, strict=True):
            heap.extend(int(w).bit_length() - 1, [Elem(s)])
        b.compress(heap, max_height=3 if k > 3 else 2)
        b.netlist.outputs = [b.heap_at_least(heap, threshold)]
        assert b.netlist.depth() <= 6
        out = simulate(b.netlist.sweep(), x).ravel()
        assert (out == (value >= threshold)).all(), (k, threshold)


@pytest.mark.parametrize("k", [3, 4, 5, 6])
@pytest.mark.parametrize(("absorb", "clamp"), [(False, False), (True, False), (False, True)])
def test_comparator_refinements_are_exact_for_every_threshold(k, absorb, clamp):
    # Tall, uneven columns (6 bits of weight 1, 5 of weight 2, one inverted, 3 of weight 4,
    # 2 of weight 8) so that steps have free inputs for next-column bits to fill.
    n_inputs = 16
    weights = np.array([1] * 6 + [2] * 5 + [4] * 3 + [8] * 2)
    x = all_vectors(n_inputs).astype(np.int64)
    value = x @ weights + (1 - 2 * x[:, 6]) * 2
    for threshold in range(-1, 46, 2):
        b = _Builder(k=k, num_inputs=n_inputs)
        heap = BitHeap()
        for i, w in enumerate(weights):
            s = b.netlist.input_signal(i)
            heap.extend(int(w).bit_length() - 1, [Elem(s, 1, 0) if i == 6 else Elem(s)])
        b.compress(heap, max_height=min(k - 1, 4))
        b.netlist.outputs = [b.heap_at_least(heap, threshold, absorb=absorb, clamp=clamp)]
        out = simulate(b.netlist.sweep(), x).ravel()
        assert (out == (value >= threshold)).all(), (k, absorb, clamp, threshold)


@pytest.mark.parametrize("k", [3, 4, 5, 6])
@pytest.mark.parametrize("plan", ["shallow", "cheap"])
@pytest.mark.parametrize(("max_height", "clamp"), [(0, True), (2, True), (4, True), (4, False)])
def test_planned_chain_is_exact_for_every_threshold(k, plan, max_height, clamp):
    # Uneven columns (6 bits of weight 1, 5 of weight 2, one inverted, 3 of weight 4, 2 of
    # weight 8), compressed or handed to the chain as they are (max_height 0), so that plans
    # use whole columns, next-column bits and partial columns.
    n_inputs = 16
    weights = np.array([1] * 6 + [2] * 5 + [4] * 3 + [8] * 2)
    x = all_vectors(n_inputs).astype(np.int64)
    value = x @ weights + (1 - 2 * x[:, 6]) * 2
    for threshold in range(-1, 46, 2):
        b = _Builder(k=k, num_inputs=n_inputs)
        heap = BitHeap()
        for i, w in enumerate(weights):
            s = b.netlist.input_signal(i)
            heap.extend(int(w).bit_length() - 1, [Elem(s, 1, 0) if i == 6 else Elem(s)])
        if max_height:
            b.compress(heap, max_height=min(k - 1, max_height))
        b.netlist.outputs = [b.heap_at_least(heap, threshold, clamp=clamp, plan=plan)]
        out = simulate(b.netlist.sweep(), x).ravel()
        assert (out == (value >= threshold)).all(), (k, plan, max_height, clamp, threshold)


@pytest.mark.parametrize(("k", "n", "luts", "depth"), [(6, 8, 3, 2), (4, 6, 3, 2)])
def test_partial_column_steps_shrink_small_neurons(k, n, luts, depth):
    # popcount >= n / 2 over a few more bits than fit in a LUT. The first step takes k bits
    # of the column and keeps only the carry values that can still change the output
    # (8 bits at k=6: six bits clamp to {<=1, 2, 3, >=4}, two carry bits), so no counter is
    # needed: 8 inputs at k=6 took 4 LUTs and 6 inputs at k=4 took 4 before.
    x = all_vectors(n)
    b = _Builder(k=k, num_inputs=n)
    t = n // 2 + (n % 2)
    out = b.neuron([(b.netlist.input_signal(i), 1) for i in range(n)], t)
    b.netlist.outputs = [out]
    nl = b.netlist.sweep()
    assert (simulate(nl, x).ravel() == (x.sum(axis=1) >= t)).all()
    assert (nl.lut_count(), nl.depth()) == (luts, depth)


@pytest.mark.parametrize("k", [4, 6])
def test_planned_chains_never_make_a_neuron_worse(k):
    # The greedy chain stays one of the candidates, so each neuron's cheapest version and its
    # shallowest version are never worse than with the greedy chain alone; across widths and
    # thresholds many get smaller.
    rng = np.random.default_rng(7)
    greedy_total = planned_total = 0
    for n in range(k + 1, 3 * k + 4):
        w = rng.choice(np.array([-1, 1]), size=n)
        for t in (1, int(round(0.5 * w.sum())), n - 2):
            b = _Builder(k=k, num_inputs=n)
            sig = [b.netlist.input_signal(i) for i in range(n)]
            elems = [Elem(s, 1, 0) if wi < 0 else Elem(s) for s, wi in zip(sig, w, strict=True)]
            lt = t + int((w < 0).sum())

            def prepare(h, elems=elems, b=b):
                heap = BitHeap(cols={0: list(elems)})
                if h:
                    b.compress(heap, max_height=h)
                return heap

            tried = [r for h in range(k + 1) for r in b.trial_chains(lambda h=h: prepare(h), lt)]
            greedy = [r for r in tried if r[2] == "greedy"]

            def cheapest(rs):
                return min((r[0], r[1]) for r in rs)

            def shallowest(rs):
                return min((r[1], r[0]) for r in rs)

            assert cheapest(tried) <= cheapest(greedy), (n, t)
            assert shallowest(tried) <= shallowest(greedy), (n, t)
            greedy_total += cheapest(greedy)[0]
            planned_total += cheapest(tried)[0]
    assert planned_total < greedy_total


def test_carry_is_clamped_to_values_that_matter():
    # q' = floor((q + v - 1) / 2) for q in [0, 1], v in [0, 6], t_rest = 9 (t_low = 1):
    # q' in [-1, 3], three carry bits. The next step needs q' + rest >= 4 with rest in
    # [0, 1], so q' <= 2 all behave like 2: q' in [2, 3] is one carry bit.
    lo, hi = _Builder._carry_range(q_lo=0, q_hi=1, v_hi=6, t_rest=9, delta=1, r_hi=1)
    assert (lo, hi) == (2, 3)
    # With plenty left to add, nothing is clamped.
    assert _Builder._carry_range(0, 1, 6, 9, 1, r_hi=20) == (-1, 3)


@pytest.mark.parametrize("k", [4, 6])
def test_comparator_refinements_never_cost_luts(k):
    # Across neuron widths, the refined comparator chain removes heap bits or carry bits for
    # free, so each neuron's cheapest version never gets larger, and many get smaller.
    rng = np.random.default_rng(3)
    plain = refined = 0
    for n in range(k + 1, 40, 2):
        w = rng.choice(np.array([-1, 1]), size=n)
        t = int(round(0.5 * w.sum())) + int(rng.integers(-1, 2))
        best = {}
        for refine in (False, True):
            b = _Builder(k=k, num_inputs=n)
            sig = [b.netlist.input_signal(i) for i in range(n)]
            elems = [Elem(s, 1, 0) if wi < 0 else Elem(s) for s, wi in zip(sig, w, strict=True)]
            lt = t + int((w < 0).sum())

            def build(h, b=b, elems=elems, lt=lt, refine=refine):
                heap = BitHeap(cols={0: list(elems)})
                b.compress(heap, max_height=h)
                return b.heap_at_least(heap, lt, absorb=refine, clamp=refine)

            best[refine] = min(b.trial(lambda h=h: build(h))[0] for h in range(2, k + 1))
        assert best[True] <= best[False], n
        plain, refined = plain + best[False], refined + best[True]
    assert refined < plain


@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_wide_neuron_matches_reference_for_every_threshold(k):
    rng = np.random.default_rng(5)
    w = rng.choice(np.array([-1, 1]), size=16)
    x = all_vectors(16)
    for threshold in range(int(w.clip(max=0).sum()) - 1, int(w.clip(min=0).sum()) + 2):
        net = TernaryNetwork((TernaryLayer(w[None, :], np.array([threshold])),))
        nl = compile_network(net, k=k)
        assert (simulate(nl, x).ravel() == net.forward(x).ravel()).all(), (k, threshold)


def test_negative_weights_cost_no_extra_luts():
    w = np.ones(24, dtype=np.int8)
    pos = compile_network(TernaryNetwork((TernaryLayer(w[None, :], np.array([12])),)))
    neg = compile_network(TernaryNetwork((TernaryLayer(-w[None, :], np.array([-12])),)))
    assert pos.lut_count() == neg.lut_count()
    assert pos.depth() == neg.depth()


def test_wide_neuron_beats_ripple_carry_adder_tree():
    # 48 inputs: the previous chunked ripple-carry compiler used 82 LUTs at depth 10.
    w = np.random.default_rng(2).choice(np.array([-1, 1]), size=48).astype(np.int8)
    net = TernaryNetwork((TernaryLayer(w[None, :], np.array([0])),))
    nl = compile_network(net)
    assert check_equivalence(net, nl, num_samples=8192).equivalent
    assert nl.lut_count() <= 45 and nl.depth() <= 6


# -- end to end --------------------------------------------------------------------------------


@pytest.mark.parametrize("share", [False, True])
def test_network_is_never_deeper_than_its_shallowest_build(share):
    # The second pass may give neurons off the critical path deeper, cheaper versions, but
    # never more depth than building every neuron in its shallowest version.
    for k, seed in ((6, 0), (4, 1)):
        net = random_network([96, 48, 24, 6], 0.7, np.random.default_rng(seed))
        shallow, _ = _compile(net, k, True, True, share, {})
        shallow_nl = shallow.netlist.sweep()
        nl = compile_network(net, k=k, share_counters=share)
        assert check_equivalence(net, nl, num_samples=2048).equivalent
        assert nl.depth() <= shallow_nl.depth()
        assert nl.lut_count() < shallow_nl.lut_count()


def test_neuron_respects_its_required_level():
    w = np.random.default_rng(4).choice(np.array([-1, 1]), size=40)
    b = _Builder(k=6, num_inputs=40)
    terms = [(b.netlist.input_signal(i), int(v)) for i, v in enumerate(w)]
    tried = {}
    cheapest = b.trial(lambda: b.neuron(terms, 0, costs=tried))
    shallowest = b.trial(lambda: b.neuron(terms, 0, shallowest=True, costs=tried))
    assert shallowest[1] <= cheapest[1] and shallowest[0] >= cheapest[0]
    capped = b.trial(lambda: b.neuron(terms, 0, max_level=shallowest[1], costs=tried))
    assert capped[1] <= shallowest[1] and capped[0] <= shallowest[0]
    assert len(tried) == 1  # the trial results were reused


def test_small_neuron_compiles_to_one_lut():
    net = TernaryNetwork((TernaryLayer(np.array([[1, 0, -1, 1, 0, 0, 0, 1]]), np.array([1])),))
    nl = compile_network(net, k=6)
    assert nl.lut_count() == 1
    assert len(nl.luts[0].inputs) == 4  # zero weights dropped
    assert check_equivalence(net, nl).equivalent


def test_all_zero_neuron_becomes_constant():
    net = TernaryNetwork((TernaryLayer(np.zeros((2, 5), dtype=np.int8), np.array([0, 1])),))
    nl = compile_network(net)
    assert nl.lut_count() == 0
    assert nl.outputs == [CONST1, CONST0]


@pytest.mark.parametrize("k", [3, 4, 5, 6])
@pytest.mark.parametrize("sparsity", [0.0, 0.5, 0.9])
@pytest.mark.parametrize("seed", range(3))
def test_compiled_circuit_matches_reference_exhaustively(k, sparsity, seed):
    net = random_network([14, 20, 6], sparsity, np.random.default_rng(seed))
    for prune in (True, False):
        nl = compile_network(net, k=k, prune_zeros=prune)
        result = check_equivalence(net, nl)
        assert result.exhaustive and result.equivalent, (k, sparsity, seed, prune)


def test_large_network_matches_reference_on_samples():
    net = random_network([200, 64, 10], 0.85, np.random.default_rng(7))
    result = check_equivalence(net, compile_network(net), num_samples=2048)
    assert result.equivalent and not result.exhaustive


def test_sparsity_reduces_luts():
    net = random_network([64, 32, 8], 0.8, np.random.default_rng(3))
    r = evaluate(net)
    assert r.equivalent
    assert r.lut_count < r.baseline_lut_count
    assert r.lut_reduction > 0.3


# -- counters shared across neurons -------------------------------------------------------------


def test_shared_groups_respect_orientation_and_disjointness():
    rng = np.random.default_rng(4)
    w = random_network([96, 40], 0.6, rng).layers[0].weights
    groups = find_shared_groups(w, 6, 6)
    assert groups
    used: dict[int, set[int]] = {}
    for g in groups:
        assert len(g.lits) == 6 and len(g.members) >= 2
        for m, orient in g.members:
            for i, wi in g.lits:
                assert w[m, i] == orient * wi != 0
                assert i not in used.setdefault(m, set())
                used[m].add(i)


def test_shared_groups_serve_the_most_constrained_neuron_first():
    # Neurons 0 and 1 can each share three literals only with neuron 2. Neuron 2 has four
    # in common with neuron 1 (inputs 1, 2, 5, 7, inverted) and three with neuron 0
    # (inputs 3, 5, 6, inverted). Taking the pair with the most common literals first
    # (1 and 2) uses up input 5 and leaves neuron 0 with nothing to share; serving neuron 0
    # first lets both pairs share a full adder.
    w = np.array(
        [
            [0, 0, 1, -1, 1, -1, 1, 0],
            [1, 1, -1, 0, 0, -1, 0, 1],
            [1, -1, 1, 1, 1, 1, -1, -1],
        ],
        dtype=np.int8,
    )
    groups = find_shared_groups(w, 3, 3)
    assert sorted(tuple(m for m, _ in g.members) for g in groups) == [(0, 2), (1, 2)]
    for g in groups:
        for m, orient in g.members:
            assert all(w[m, i] == orient * wi for i, wi in g.lits)
    x = all_vectors(8)
    for t in range(-2, 3):
        net = TernaryNetwork((TernaryLayer(w, np.array([t, -t, t])),))
        nl = compile_network(net, k=4)
        assert (simulate(nl, x) == net.forward(x)).all(), t


def test_shared_group_keeps_widely_held_literals_free():
    # Neurons 0 and 1 share inputs 0-3, but input 0 is also what neurons 2 and 3 need to
    # fill a group of three, so the first group takes inputs 1-3 and leaves 0 to them.
    w = np.array(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 1, 1],
        ],
        dtype=np.int8,
    )
    groups = find_shared_groups(w, 3, 0)
    found = sorted((sorted(i for i, _ in g.lits), [m for m, _ in g.members]) for g in groups)
    assert found == [([0, 4, 5], [2, 3]), ([1, 2, 3], [0, 1])]


def test_inverted_member_reads_shared_counter_exactly():
    # Neuron 1 has neuron 0's first eight weights negated: they share one counter
    # in opposite orientations. Exhaustive over all 2^14 inputs and every threshold.
    a = np.array([1, -1, 1, 1, -1, 1, 1, -1, 1, 0, 1, -1, 0, 1], dtype=np.int8)
    b = np.concatenate([-a[:8], np.array([1, 1, -1, 0, 1, 1], dtype=np.int8)])
    w = np.stack([a, b])
    groups = find_shared_groups(w, 6, 6)
    assert len(groups) == 1 and sorted(o for _, o in groups[0].members) == [-1, 1]
    x = all_vectors(14)
    for t0 in range(-3, 7):
        for t1 in range(-4, 5, 2):
            net = TernaryNetwork((TernaryLayer(w, np.array([t0, t1])),))
            nl = compile_network(net)
            assert (simulate(nl, x) == net.forward(x)).all(), (t0, t1)


@pytest.mark.parametrize("k", [4, 6])
def test_sharing_saves_luts_without_deepening(k):
    for seed in range(3):
        net = random_network([64, 24, 8], 0.6, np.random.default_rng(seed))
        shared = compile_network(net, k=k)
        alone = compile_network(net, k=k, share_counters=False)
        assert check_equivalence(net, shared, num_samples=2048).equivalent
        assert shared.depth() <= alone.depth(), (k, seed)
        assert shared.lut_count() < alone.lut_count(), (k, seed)
