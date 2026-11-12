"""The formal proof must be sound: it may only call a circuit proven when it is equivalent."""

import dataclasses
import itertools

import numpy as np
import pytest

from dark_algorithms.ternary_lut import (
    Lut,
    TernaryLayer,
    TernaryNetwork,
    check_equivalence,
    compile_network,
    prove_equivalence,
    random_network,
    simulate,
)
from dark_algorithms.ternary_lut.prove import Bdd, BudgetExceeded


def all_vectors(n: int) -> np.ndarray:
    return np.array(list(itertools.product((0, 1), repeat=n)), dtype=np.uint8)


def flip(netlist, lut_index: int, bit: int):
    """A copy of the netlist with one truth-table bit of one LUT flipped."""
    lut = netlist.luts[lut_index]
    luts = list(netlist.luts)
    luts[lut_index] = Lut(lut.inputs, lut.table ^ (1 << bit))
    return dataclasses.replace(netlist, luts=luts)


# -- BDD manager -------------------------------------------------------------------------------


def bdd_eval(bdd: Bdd, f: int, bits) -> int:
    while f > 1:
        f = bdd._hi[f] if bits[bdd._var[f]] else bdd._lo[f]
    return f


@pytest.mark.parametrize("seed", range(5))
def test_threshold_bdd_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    weights = [int(w) for w in rng.integers(-4, 5, size=7)]
    theta = int(rng.integers(-6, 7))
    bdd = Bdd(7)
    f = bdd.threshold(weights, theta)
    for bits in all_vectors(7):
        assert bdd_eval(bdd, f, bits) == int(np.dot(weights, bits) >= theta)


@pytest.mark.parametrize("seed", range(5))
def test_lut_bdd_matches_truth_table(seed):
    rng = np.random.default_rng(seed)
    bdd = Bdd(5)
    args = [bdd.var(v) for v in range(5)]
    args[2] = bdd.xor(args[2], args[4])  # a non-variable argument
    table = int(rng.integers(0, 1 << 16))
    f = bdd.lut(table, args[:4])
    for bits in all_vectors(5):
        a = [int(b) for b in (bits[0], bits[1], bits[2] ^ bits[4], bits[3])]
        idx = sum(b << j for j, b in enumerate(a))
        assert bdd_eval(bdd, f, bits) == (table >> idx) & 1


def test_bdd_is_canonical():
    bdd = Bdd(3)
    x, y, z = (bdd.var(v) for v in range(3))
    left = bdd.ite(x, bdd.ite(y, 1, z), z)  # x & y | z
    right = bdd.ite(z, 1, bdd.ite(x, y, 0))
    assert left == right


def test_bdd_budget():
    bdd = Bdd(20, max_nodes=50)
    with pytest.raises(BudgetExceeded):
        bdd.threshold([1] * 20, 10)


# -- proofs ------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [4, 6])
@pytest.mark.parametrize("seed", range(3))
def test_compiled_networks_are_proven(k, seed):
    net = random_network([40, 24, 8], 0.6, np.random.default_rng(seed))
    nl = compile_network(net, k=k)
    result = prove_equivalence(net, nl)
    assert result.proven, result.reason
    assert result.neurons_proven == result.neurons_total == 32


def test_wide_shared_network_is_proven():
    """Wide neurons with shared counters: the case where a plain BDD runs out of memory."""
    net = random_network([200, 48, 8], 0.65, np.random.default_rng(5))
    result = prove_equivalence(net, compile_network(net))
    assert result.proven, result.reason


@pytest.mark.parametrize("seed", range(4))
def test_proof_agrees_with_exhaustive_simulation_under_mutation(seed):
    """Flip LUT bits: whenever the proof still succeeds, the circuit must really be correct."""
    net = random_network([14, 12, 6], 0.4, np.random.default_rng(seed))
    nl = compile_network(net, k=4)
    x = all_vectors(14)
    rng = np.random.default_rng(100 + seed)
    detected = 0
    for _ in range(40):
        n = int(rng.integers(len(nl.luts)))
        bad = flip(nl, n, int(rng.integers(1 << len(nl.luts[n].inputs))))
        really_equal = bool((simulate(bad, x) == net.forward(x)).all())
        result = prove_equivalence(net, bad)
        if result.proven:
            assert really_equal
        if not really_equal:
            detected += 1
            assert not result.proven
    assert detected > 10


def test_single_layer_bug_gives_a_real_counterexample():
    net = random_network([12, 6], 0.3, np.random.default_rng(3))
    nl = compile_network(net, k=4)
    x = all_vectors(12)
    rng = np.random.default_rng(0)
    while True:
        n = int(rng.integers(len(nl.luts)))
        bad = flip(nl, n, int(rng.integers(1 << len(nl.luts[n].inputs))))
        if not (simulate(bad, x) == net.forward(x)).all():
            break
    result = prove_equivalence(net, bad)
    assert not result.proven
    if result.counterexample is not None:
        assert result.refuted
        v = np.array([result.counterexample], dtype=np.uint8)
        assert (simulate(bad, v) != net.forward(v)).any()


@pytest.mark.parametrize("seed", range(3))
def test_bugs_in_wide_networks_are_not_proven(seed):
    """Mutations that random simulation catches in wide neurons must never be proven."""
    net = random_network([120, 16, 4], 0.6, np.random.default_rng(seed))
    nl = compile_network(net)
    x = np.random.default_rng(seed).integers(0, 2, size=(4096, 120), dtype=np.uint8)
    rng = np.random.default_rng(200 + seed)
    caught = 0
    for _ in range(60):
        n = int(rng.integers(len(nl.luts)))
        bad = flip(nl, n, int(rng.integers(1 << len(nl.luts[n].inputs))))
        if (simulate(bad, x) != net.forward(x)).any():
            caught += 1
            assert not prove_equivalence(net, bad).proven
    assert caught > 5


def test_wrong_reference_is_not_proven():
    net = random_network([60, 20, 5], 0.5, np.random.default_rng(1))
    nl = compile_network(net)
    layers = list(net.layers)
    t = layers[0].thresholds.copy()
    t[3] += 1
    layers[0] = TernaryLayer(layers[0].weights, t)
    other = TernaryNetwork(tuple(layers))
    sampled = check_equivalence(other, nl).equivalent
    assert not prove_equivalence(other, nl).proven or sampled


def test_constant_and_merged_hidden_neurons_are_proven():
    """Hidden neurons folded to constants or to one shared signal, and a swept neuron."""
    w0 = np.array(
        [
            [1, 1, -1, 0, 0, 1],
            [1, 1, -1, 0, 0, 1],  # same function as neuron 0: one signal
            [1, -1, 0, 0, 0, 0],  # x0 - x1 >= -1 is always true: constant
            [0, 0, 0, 1, 1, 0],  # read only by a constant neuron, so swept away
        ]
    )
    w1 = np.array([[1, -1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 1, -1]])
    w2 = np.array([[1, 1, -1, 1]])
    net = TernaryNetwork(
        (
            TernaryLayer(w0, np.array([1, 1, -1, 2])),
            TernaryLayer(w1, np.array([1, 2, 1, -1])),
            TernaryLayer(w2, np.array([2])),
        )
    )
    nl = compile_network(net, k=3)
    x = all_vectors(6)
    assert (simulate(nl, x) == net.forward(x)).all()
    assert prove_equivalence(net, nl).proven


def counter_then_compare(compare_table: int, theta: int):
    """A 6:3 counter over six inputs read by one comparator LUT, and its reference neuron.

    The counter's three bits never read 7, so a comparator may differ from the threshold
    there and still be correct, as the compiler's comparator chains do since they track the
    exact range of their heap.
    """
    from dark_algorithms.ternary_lut import Netlist

    nl = Netlist(k=6, num_inputs=6)
    ins = tuple(nl.input_signal(i) for i in range(6))
    for j in range(3):
        table = sum(1 << r for r in range(64) if (bin(r).count("1") >> j) & 1)
        nl.luts.append(Lut(ins, table))
    bits = tuple(nl.first_lut_signal + j for j in range(3))
    nl.luts.append(Lut(bits, compare_table))
    nl.outputs = [nl.first_lut_signal + 3]
    nl.layer_outputs = [list(nl.outputs)]
    net = TernaryNetwork((TernaryLayer(np.ones((1, 6), dtype=np.int8), np.array([theta])),))
    return net, nl


def test_comparator_may_use_the_counters_range():
    # count >= 6 as "bits read exactly 6": wrong only for the impossible count 7.
    net, nl = counter_then_compare(1 << 6, theta=6)
    assert (simulate(nl, all_vectors(6)) == net.forward(all_vectors(6))).all()
    assert prove_equivalence(net, nl).proven


def test_compiled_comparator_using_a_shared_counters_range_is_proven():
    """Shared counters read inverted, skewed thresholds: the comparator relies on the range.

    The proof used to treat counter outputs as free bits and refuted this correct circuit.
    """
    w = np.array(
        [
            [1, 1, -1, -1, -1, -1, 1, 0, 1, -1, -1, -1, -1],
            [1, 1, -1, -1, -1, -1, 1, -1, 0, -1, 0, 0, -1],
            [0, -1, 0, 1, 1, 0, 0, 1, -1, 1, 1, 1, 1],
        ]
    )
    net = TernaryNetwork((TernaryLayer(w, np.array([4, -8, 1])),))
    nl = compile_network(net, k=4)
    x = all_vectors(13)
    assert (simulate(nl, x) == net.forward(x)).all()
    assert prove_equivalence(net, nl).proven


def test_comparator_wrong_on_a_possible_count_is_not_proven():
    # "bits read 7" never fires, but count 6 is possible: the circuit is wrong.
    net, nl = counter_then_compare(1 << 7, theta=6)
    assert not (simulate(nl, all_vectors(6)) == net.forward(all_vectors(6))).all()
    result = prove_equivalence(net, nl)
    assert not result.proven and result.refuted


def test_missing_layer_outputs_are_not_proven():
    net = random_network([10, 4], 0.3, np.random.default_rng(0))
    nl = compile_network(net)
    nl.layer_outputs = []
    result = prove_equivalence(net, nl)
    assert not result.proven and "layer outputs" in result.reason


def test_time_box():
    net = random_network([200, 48, 8], 0.65, np.random.default_rng(5))
    result = prove_equivalence(net, compile_network(net), time_box=0.0)
    assert not result.proven and "time box" in result.reason
