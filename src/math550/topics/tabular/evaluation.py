"""Evaluation helpers for paired scratch-versus-library experiments."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def validate_probability_output(
    probabilities: np.ndarray, n_samples: int
) -> np.ndarray:
    """Validate a binary ``predict_proba`` result and return its positive column."""

    values = np.asarray(probabilities, dtype=float)
    if values.shape != (n_samples, 2):
        raise ValueError(
            f"predict_proba must return shape ({n_samples}, 2), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("predict_proba returned NaN or infinite values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("predicted probabilities must lie in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("binary probability rows must sum to one")
    return values[:, 1]


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Return threshold-free, probability, and threshold-0.5 metrics."""

    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, probability),
        "average_precision": average_precision_score(y_true, probability),
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall_sensitivity": recall_score(y_true, prediction, zero_division=0),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "log_loss": log_loss(y_true, probability, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, probability),
    }


def stratified_paired_bootstrap_gap(
    y_true: np.ndarray,
    scratch_probability: np.ndarray,
    library_probability: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    n_bootstrap: int = 2_000,
    random_state: int = 550,
) -> tuple[float, float, float]:
    """Estimate scratch-minus-library gap and a paired percentile interval."""

    y_true = np.asarray(y_true, dtype=int)
    scratch_probability = np.asarray(scratch_probability, dtype=float)
    library_probability = np.asarray(library_probability, dtype=float)
    groups = [np.flatnonzero(y_true == label) for label in np.unique(y_true)]
    rng = np.random.default_rng(random_state)
    gaps = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sample = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        gaps[index] = metric(y_true[sample], scratch_probability[sample]) - metric(
            y_true[sample], library_probability[sample]
        )
    observed = metric(y_true, scratch_probability) - metric(y_true, library_probability)
    low, high = np.quantile(gaps, [0.025, 0.975])
    return float(observed), float(low), float(high)


def probability_agreement(
    scratch_probability: np.ndarray, library_probability: np.ndarray
) -> dict[str, float]:
    """Quantify predictive agreement beyond aggregate accuracy."""

    scratch_probability = np.asarray(scratch_probability, dtype=float)
    library_probability = np.asarray(library_probability, dtype=float)
    return {
        "probability_mae": float(
            np.mean(np.abs(scratch_probability - library_probability))
        ),
        "probability_correlation": float(
            np.corrcoef(scratch_probability, library_probability)[0, 1]
        ),
        "label_agreement": float(
            np.mean((scratch_probability >= 0.5) == (library_probability >= 0.5))
        ),
    }
