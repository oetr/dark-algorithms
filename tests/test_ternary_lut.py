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
from dark_algorithms.ternary_lut.compiler import BitHeap, Elem, _Builder
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
def test_compression_variants_are_exact(k, gpc, group_size):
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
        b.compress(heap, max_height=2, gpc=gpc, group_size=group_size)
        b.netlist.outputs = [b.heap_at_least(heap, threshold)]
        out = simulate(b.netlist.sweep(), x).ravel()
        assert (out == (value >= threshold)).all(), (k, gpc, group_size, threshold)


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
