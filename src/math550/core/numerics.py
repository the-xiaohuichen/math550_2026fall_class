"""Numerically stable primitives shared across teaching topics."""

from __future__ import annotations

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""

    values = np.asarray(values, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def logit(probability: float, epsilon: float = 1e-12) -> float:
    """Return a clipped log-odds value."""

    probability = float(np.clip(probability, epsilon, 1.0 - epsilon))
    return float(np.log(probability / (1.0 - probability)))


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Proximal operator for the L1 norm."""

    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)
