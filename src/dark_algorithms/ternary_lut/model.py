"""Ternary neural networks: the *input* of the LUT compiler.

A ternary network maps a vector of bits to a vector of bits. Every layer is a
set of threshold neurons with weights in {-1, 0, +1}:

    y_j = 1  if  sum_i W[j, i] * x_i >= theta_j  else 0

Inputs and activations are bits, so the whole network is a boolean function and
can be implemented exactly as a circuit of lookup tables (LUTs) with no memory
cells holding weights. Zero weights simply disappear from the circuit, which is
where sparsity pays off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TernaryLayer:
    """One layer of threshold neurons with ternary weights.

    weights: int8 array of shape (out_features, in_features), values in {-1, 0, 1}.
    thresholds: int64 array of shape (out_features,).
    """

    weights: np.ndarray
    thresholds: np.ndarray

    def __post_init__(self) -> None:
        w = np.asarray(self.weights)
        t = np.asarray(self.thresholds)
        if w.ndim != 2:
            raise ValueError(f"weights must be 2-D, got shape {w.shape}")
        if t.shape != (w.shape[0],):
            raise ValueError(f"thresholds must have shape ({w.shape[0]},), got {t.shape}")
        if not np.isin(w, (-1, 0, 1)).all():
            raise ValueError("weights must be ternary: values in {-1, 0, 1}")
        object.__setattr__(self, "weights", w.astype(np.int8))
        object.__setattr__(self, "thresholds", t.astype(np.int64))

    @property
    def in_features(self) -> int:
        return int(self.weights.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.weights.shape[0])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Reference forward pass. x: (batch, in_features) of {0, 1}. Returns uint8 bits."""
        pre = x.astype(np.int64) @ self.weights.T.astype(np.int64)
        return (pre >= self.thresholds).astype(np.uint8)


@dataclass(frozen=True)
class TernaryNetwork:
    """A feed-forward stack of ternary layers. This is the reference semantics."""

    layers: tuple[TernaryLayer, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("a network needs at least one layer")
        for prev, nxt in zip(self.layers, self.layers[1:], strict=False):
            if prev.out_features != nxt.in_features:
                raise ValueError(
                    f"layer size mismatch: {prev.out_features} outputs feed "
                    f"{nxt.in_features} inputs"
                )

    @property
    def num_inputs(self) -> int:
        return self.layers[0].in_features

    @property
    def num_outputs(self) -> int:
        return self.layers[-1].out_features

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Reference forward pass over a batch of bit vectors, shape (batch, num_inputs)."""
        h = np.asarray(x, dtype=np.uint8)
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def weight_count(self) -> int:
        return int(sum(layer.weights.size for layer in self.layers))

    def nonzero_weight_count(self) -> int:
        return int(sum(np.count_nonzero(layer.weights) for layer in self.layers))

    def sparsity(self) -> float:
        """Fraction of weights that are zero."""
        return 1.0 - self.nonzero_weight_count() / self.weight_count()


def random_network(
    layer_sizes: list[int] | tuple[int, ...],
    sparsity: float,
    rng: np.random.Generator,
) -> TernaryNetwork:
    """Draw a random ternary network.

    layer_sizes: [num_inputs, hidden..., num_outputs].
    sparsity: probability that a weight is zero; non-zero weights are +1 or -1 with equal odds.
    Thresholds are centred on each neuron's expected pre-activation for uniform random input
    bits, so activations are roughly balanced and the network does not collapse to constants.
    """
    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes needs at least an input and an output size")
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    layers = []
    for fan_in, fan_out in zip(layer_sizes, layer_sizes[1:], strict=False):
        nonzero = rng.random((fan_out, fan_in)) >= sparsity
        signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=(fan_out, fan_in))
        weights = (nonzero * signs).astype(np.int8)
        expected = 0.5 * weights.sum(axis=1)
        thresholds = np.round(expected).astype(np.int64) + rng.integers(-1, 2, size=fan_out)
        layers.append(TernaryLayer(weights=weights, thresholds=thresholds))
    return TernaryNetwork(layers=tuple(layers))
