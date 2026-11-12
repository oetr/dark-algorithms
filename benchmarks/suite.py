"""The fixed benchmark suite. Results are only comparable across commits if this stays fixed.

Add new cases rather than changing existing ones; the regression gate compares cases by name.

Most cases are random networks, drawn from their seed. A case with a `network_file` is a
network trained on a real task instead, stored under benchmarks/networks/ (see
train_digits.py); its layer sizes and sparsity are those of the stored network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dark_algorithms.ternary_lut import TernaryLayer, TernaryNetwork, random_network

NETWORKS_DIR = Path(__file__).resolve().parent / "networks"


@dataclass(frozen=True)
class Case:
    name: str
    layer_sizes: tuple[int, ...]
    sparsity: float
    seed: int
    k: int = 6
    network_file: str | None = None

    def network(self) -> TernaryNetwork:
        """The case's network: loaded from its file, or drawn at random from its seed."""
        if self.network_file is None:
            return random_network(self.layer_sizes, self.sparsity, np.random.default_rng(self.seed))
        net = load_network(NETWORKS_DIR / self.network_file)
        sizes = (net.num_inputs, *(layer.out_features for layer in net.layers))
        if sizes != tuple(self.layer_sizes):
            raise ValueError(f"{self.name}: stored network has layer sizes {sizes}")
        return net


def load_network(path: Path) -> TernaryNetwork:
    """Read a network written by train_digits.py: one +/0/- string per neuron and layer."""
    signs = {"+": 1, "0": 0, "-": -1}
    data = json.loads(path.read_text())
    layers = []
    for layer in data["layers"]:
        weights = np.array([[signs[c] for c in row] for row in layer["weights"]], dtype=np.int8)
        layers.append(TernaryLayer(weights=weights, thresholds=np.array(layer["thresholds"])))
    return TernaryNetwork(layers=tuple(layers))


SUITE: tuple[Case, ...] = (
    Case("tiny-exhaustive-s50", (16, 16, 8), 0.5, seed=1),
    Case("mlp-64-s70", (64, 64, 32, 10), 0.7, seed=1),
    Case("mlp-256-s90", (256, 128, 64, 10), 0.9, seed=1),
    Case("mlp-256-s95", (256, 128, 64, 10), 0.95, seed=2),
    Case("mlp-256-s70", (256, 128, 64, 10), 0.7, seed=3),
    Case("mlp-256-s90-k4", (256, 128, 64, 10), 0.9, seed=1, k=4),
    # Trained on handwritten digits; same shape and sparsity as mlp-64-s70, so the two show
    # how a trained network differs from a random one.
    Case("digits-64-s70", (64, 64, 32, 10), 0.7, seed=0, network_file="digits-64-s70.json"),
)
