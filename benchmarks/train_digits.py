"""Train the ternary network behind the benchmark's trained case on handwritten digits.

    python benchmarks/train_digits.py [--output benchmarks/networks/digits-64-s70.json]

Every other benchmark network is random: independent weights, a fixed share of zeros per
row and thresholds centred on each neuron's expected sum. A network trained on a real task
looks different: some inputs matter far more than others (in 8x8 digit images the border
pixels are almost always blank), neurons have very different fan-ins, thresholds sit far
from the centre, and some neurons end up constant or duplicated. This script produces such
a network so the compiler is measured on one.

Task: scikit-learn's bundled handwritten digits (1,797 images of 8x8 pixels, no download).
Each pixel becomes one input bit (ink >= 8 of 16). The network is a ternary threshold MLP,
64 -> 64 -> 32 -> 10, the same model the compiler takes: weights in {-1, 0, +1}, every
neuron fires when its weighted sum reaches an integer threshold. The ten outputs are
one-vs-rest digit detectors; an image counts as classified correctly when exactly its own
digit's output is 1.

Training is the usual straight-through recipe for ternary and binary networks: real latent
weights are ternarized in the forward pass (the 70% smallest of each layer become 0, the rest
their sign), activations are hard thresholds, and gradients pass through both as if they
were the identity (clipped). Thresholds are learned as real biases and rounded to integers in
every forward pass, so the trained network is exactly the one evaluated. Everything is seeded;
the result is written once and committed, so the benchmark needs neither scikit-learn nor
this script (install the `train` extra to rerun it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "networks" / "digits-64-s70.json"
LAYER_SIZES = (64, 64, 32, 10)
SPARSITY = 0.7
PIXEL_THRESHOLD = 8  # of 16: a pixel is "ink" from half intensity on


def load_data(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.datasets import load_digits

    digits = load_digits()
    x = (digits.data >= PIXEL_THRESHOLD).astype(np.float64)
    y = digits.target.astype(np.int64)
    order = np.random.default_rng(seed).permutation(len(x))
    split = int(0.8 * len(x))
    train, test = order[:split], order[split:]
    return x[train], y[train], x[test], y[test]


def ternarize(w: np.ndarray, sparsity: float) -> np.ndarray:
    """Zero the `sparsity` share of smallest magnitudes, keep the sign of the rest."""
    cut = np.quantile(np.abs(w), sparsity)
    return np.where(np.abs(w) > cut, np.sign(w), 0.0)


class Model:
    def __init__(self, sizes: tuple[int, ...], sparsity: float, rng: np.random.Generator):
        self.sparsity = sparsity
        self.w = [rng.normal(0, 1, (o, i)) for i, o in zip(sizes, sizes[1:], strict=False)]
        self.b = [np.zeros(o) for o in sizes[1:]]

    def ternary(self) -> list[np.ndarray]:
        return [ternarize(w, self.sparsity) for w in self.w]

    def thresholds(self) -> list[np.ndarray]:
        # A neuron fires when sum(w * x) + b >= 0, i.e. sum(w * x) >= ceil(-b); sums are
        # integers, so rounding the bias is the same as an integer threshold.
        return [np.ceil(-b - 1e-9).astype(np.int64) for b in self.b]

    def forward(self, x: np.ndarray) -> tuple[list, list[np.ndarray]]:
        """Exact ternary forward pass; also returns what backward needs."""
        tape, h = [], x
        for w, t in zip(self.ternary(), self.thresholds(), strict=True):
            pre = h @ w.T - t  # >= 0 fires
            tape.append((h, w, pre))
            h = (pre >= 0).astype(np.float64)
        return tape, h

    def step(self, x: np.ndarray, target: np.ndarray, lr: float) -> float:
        """One straight-through gradient step on a squared hinge loss over the outputs."""
        tape, _ = self.forward(x)
        _, _, pre = tape[-1]
        scale = [np.sqrt(np.count_nonzero(w, axis=1).clip(1)) for _, w, _ in tape]
        # Output pre-activations should be >= +1 for the true digit and <= -1 for the others.
        sign = 2 * target - 1
        margin = 1 - sign * (pre + 0.5) / scale[-1]
        loss = float(np.mean(np.maximum(margin, 0) ** 2))
        grad = -2 * np.maximum(margin, 0) * sign / scale[-1] / len(x)
        for li in range(len(tape) - 1, -1, -1):
            h, w, pre = tape[li]
            if li < len(tape) - 1:
                # Straight-through: the step's gradient is passed where the scaled sum is
                # near the threshold.
                grad = grad * (np.abs((pre + 0.5) / scale[li]) <= 1.0)
            self.w[li] -= lr * (grad.T @ h)
            self.b[li] -= lr * grad.sum(axis=0)
            grad = grad @ w
        return loss


def accuracy(model: Model, x: np.ndarray, y: np.ndarray) -> float:
    _, out = model.forward(x)
    return float(np.mean((out == np.eye(10)[y]).all(axis=1)))


def accuracy_of_network(network, seed: int) -> tuple[float, float]:
    """Train and test accuracy of a stored TernaryNetwork on the split `seed` defines."""
    xtr, ytr, xte, yte = load_data(seed)

    def acc(x: np.ndarray, y: np.ndarray) -> float:
        out = network.forward(x.astype(np.uint8))
        return float(np.mean((out == np.eye(10)[y]).all(axis=1)))

    return acc(xtr, ytr), acc(xte, yte)


def train(seed: int = 0, epochs: int = 300, lr: float = 0.2) -> tuple[Model, float, float]:
    xtr, ytr, xte, yte = load_data(seed)
    rng = np.random.default_rng(seed)
    model = Model(LAYER_SIZES, SPARSITY, rng)
    target = np.eye(10)[ytr]
    best = (-1.0, None)
    for epoch in range(epochs):
        order = rng.permutation(len(xtr))
        for s in range(0, len(order), 32):
            idx = order[s : s + 32]
            model.step(xtr[idx], target[idx], lr * (0.3 if epoch > 0.7 * epochs else 1.0))
        acc = accuracy(model, xtr, ytr)
        if acc > best[0]:
            best = (acc, ([w.copy() for w in model.w], [b.copy() for b in model.b]))
    model.w, model.b = best[1]
    return model, accuracy(model, xtr, ytr), accuracy(model, xte, yte)


def to_json(model: Model, train_acc: float, test_acc: float, seed: int) -> dict:
    chars = {-1: "-", 0: "0", 1: "+"}
    layers = []
    for w, t in zip(model.ternary(), model.thresholds(), strict=True):
        rows = ["".join(chars[int(v)] for v in row) for row in w.astype(np.int64)]
        layers.append({"weights": rows, "thresholds": [int(v) for v in t]})
    return {
        "task": "scikit-learn handwritten digits, 8x8 pixels as bits (ink >= 8 of 16)",
        "layer_sizes": list(LAYER_SIZES),
        "target_sparsity": SPARSITY,
        "seed": seed,
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "accuracy_rule": "exactly the true digit's output is 1",
        "weights_encoding": "one string per neuron, one character per input: + 0 -",
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    model, train_acc, test_acc = train(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_json(model, train_acc, test_acc, args.seed), indent=1))
    print(f"train accuracy {train_acc:.1%}, test accuracy {test_acc:.1%} -> {args.output}")


if __name__ == "__main__":
    main()
