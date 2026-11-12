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
from dark_algorithms.ternary_lut.compiler import _Builder, bits_for_range
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


@pytest.mark.parametrize(
    "lo,hi,width", [(0, 0, 1), (-1, 0, 1), (0, 1, 2), (-6, 6, 4), (-8, 7, 4), (-9, 7, 5)]
)
def test_bits_for_range(lo, hi, width):
    assert bits_for_range(lo, hi) == width


def test_lut_builder_folds_constants_and_shares_identical_luts():
    b = _Builder(k=4, num_inputs=2)
    s0, s1 = b.netlist.input_signal(0), b.netlist.input_signal(1)
    assert b.lut((s0, s1), lambda v: 0) == CONST0
    assert b.lut((s0, s1), lambda v: 1) == CONST1
    first = b.lut((s0, s1), lambda v: v[0] & v[1])
    again = b.lut((s0, s1), lambda v: v[0] & v[1])
    assert first == again and b.netlist.lut_count() == 1


def test_adder_is_exact_over_all_inputs():
    b = _Builder(k=6, num_inputs=6)
    sig = [b.netlist.input_signal(i) for i in range(6)]
    a = b.chunk_sum([(sig[0], 1), (sig[1], 1), (sig[2], -1)], offset=-1)
    c = b.chunk_sum([(sig[3], -1), (sig[4], -1), (sig[5], 1)], offset=2)
    total = b.add(a, c)
    b.netlist.outputs = list(total.bits)
    x = all_vectors(6)
    out = simulate(b.netlist, x).astype(np.int64)
    width = len(total.bits)
    raw = (out << np.arange(width)).sum(axis=1)
    value = np.where(raw >= 1 << (width - 1), raw - (1 << width), raw)
    weights = np.array([1, 1, -1, -1, -1, 1])
    assert (value == x @ weights + 1).all()


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


@pytest.mark.parametrize("k", [3, 4, 6])
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


# -- bit-heap neurons --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [3, 4, 6])
@pytest.mark.parametrize("fan_in", [7, 9, 12, 16])
def test_wide_neuron_matches_reference_for_every_threshold(k, fan_in):
    rng = np.random.default_rng(fan_in * 10 + k)
    weights = rng.choice(np.array([-1, 1], dtype=np.int8), size=(1, fan_in))
    lo, hi = int(np.minimum(weights, 0).sum()), int(np.maximum(weights, 0).sum())
    x = all_vectors(fan_in)
    for t in range(lo - 1, hi + 2):
        net = TernaryNetwork((TernaryLayer(weights, np.array([t])),))
        nl = compile_network(net, k=k)
        assert (simulate(nl, x) == net.forward(x)).all(), (k, fan_in, t)


def test_constant_and_duplicate_inputs_are_folded():
    # Layer 1: neuron 0 is constant 1, neurons 1 and 2 are identical (shared by hashing).
    w1 = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, -1]], dtype=np.int8)
    layer1 = TernaryLayer(w1, np.array([0, 2, 2, 0]))
    # Layer 2: +1 and -1 on the two identical neurons cancel; the constant shifts the threshold.
    w2 = np.array([[1, 1, -1, 1]], dtype=np.int8)
    net = TernaryNetwork((layer1, TernaryLayer(w2, np.array([2]))))
    nl = compile_network(net)
    assert check_equivalence(net, nl).equivalent
    # Output is just "neuron 3 fires", i.e. x2 >= x3: one LUT reading two inputs.
    assert nl.lut_count() == 1 and len(nl.luts[0].inputs) == 2


def test_wide_popcount_is_small_and_shallow():
    # A 36-input majority at k=6: six (6;3) counters, a second stage, then a short carry chain.
    net = TernaryNetwork((TernaryLayer(np.ones((1, 36), dtype=np.int8), np.array([18])),))
    nl = compile_network(net, k=6)
    assert check_equivalence(net, nl, num_samples=8192).equivalent
    assert nl.lut_count() <= 32
    assert nl.depth() <= 5
