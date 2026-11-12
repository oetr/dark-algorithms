"""The benchmark suite: every case builds the network it names, trained cases included."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from suite import NETWORKS_DIR, SUITE, Case, load_network  # noqa: E402
from train_digits import accuracy_of_network, ternarize  # noqa: E402


def test_case_names_are_unique():
    assert len({c.name for c in SUITE}) == len(SUITE)


@pytest.mark.parametrize("case", SUITE, ids=lambda c: c.name)
def test_every_case_builds_its_network(case):
    net = case.network()
    sizes = (net.num_inputs, *(layer.out_features for layer in net.layers))
    assert sizes == case.layer_sizes
    assert abs(net.sparsity() - case.sparsity) < 0.02


def test_random_cases_are_reproducible():
    case = next(c for c in SUITE if c.network_file is None)
    a, b = case.network(), case.network()
    for la, lb in zip(a.layers, b.layers, strict=True):
        assert (la.weights == lb.weights).all() and (la.thresholds == lb.thresholds).all()


def test_trained_case_is_a_trained_network():
    """Unlike the random cases: unused inputs, uneven fan-in, off-centre thresholds."""
    case = next(c for c in SUITE if c.network_file)
    first = case.network().layers[0]
    fan_in = np.count_nonzero(first.weights, axis=1)
    assert (np.count_nonzero(first.weights, axis=0) == 0).sum() >= 8  # blank border pixels
    assert fan_in.max() - fan_in.min() > 20
    centre = 0.5 * first.weights.sum(axis=1)
    assert np.abs(first.thresholds - centre).mean() > 1.5


def test_stored_network_round_trips_and_rejects_wrong_shapes(tmp_path):
    case = next(c for c in SUITE if c.network_file)
    stored = json.loads((NETWORKS_DIR / case.network_file).read_text())
    net = load_network(NETWORKS_DIR / case.network_file)
    assert [layer.out_features for layer in net.layers] == stored["layer_sizes"][1:]
    wrong = Case("wrong", (64, 10), 0.7, seed=0, network_file=case.network_file)
    with pytest.raises(ValueError, match="layer sizes"):
        wrong.network()


def test_ternarize_keeps_the_largest_share_with_their_signs():
    w = np.array([[0.1, -2.0, 0.3, 4.0, -0.2], [1.5, -0.05, 0.0, -3.0, 0.25]])
    t = ternarize(w, 0.6)
    assert np.count_nonzero(t) == 4
    assert t.tolist() == [[0, -1, 0, 1, 0], [1, 0, 0, -1, 0]]


def test_stored_accuracy_is_what_the_stored_network_achieves():
    pytest.importorskip("sklearn")
    case = next(c for c in SUITE if c.network_file)
    stored = json.loads((NETWORKS_DIR / case.network_file).read_text())
    train_acc, test_acc = accuracy_of_network(case.network(), stored["seed"])
    assert train_acc == pytest.approx(stored["train_accuracy"], abs=1e-4)
    assert test_acc == pytest.approx(stored["test_accuracy"], abs=1e-4)
