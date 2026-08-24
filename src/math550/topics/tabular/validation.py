"""Input validation shared by tabular teaching estimators."""

from __future__ import annotations

import numpy as np


def check_binary_inputs(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a finite numeric matrix and a response containing both 0 and 1."""

    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=float).reshape(-1)
    if X_array.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if len(X_array) != len(y_array):
        raise ValueError("X and y must contain the same number of rows")
    if not np.isfinite(X_array).all() or not np.isfinite(y_array).all():
        raise ValueError("Scratch estimators require finite numeric inputs")
    if not np.array_equal(np.unique(y_array), np.array([0.0, 1.0])):
        raise ValueError("y must contain both binary labels 0 and 1")
    return X_array, y_array


def check_predictors(X: np.ndarray, n_features: int) -> np.ndarray:
    """Validate prediction data against a fitted tabular feature count."""

    X_array = np.asarray(X, dtype=float)
    if X_array.ndim != 2 or X_array.shape[1] != n_features:
        raise ValueError(f"X must have shape (n_samples, {n_features})")
    if not np.isfinite(X_array).all():
        raise ValueError("X must contain only finite numeric values")
    return X_array
