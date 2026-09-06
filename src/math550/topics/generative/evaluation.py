"""Distributional diagnostics for low-dimensional classroom generators."""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist, pdist
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def frechet_gaussian_distance(real: np.ndarray, generated: np.ndarray) -> float:
    """Fréchet distance between Gaussian fits in the declared feature space."""

    real = np.asarray(real, dtype=float)
    generated = np.asarray(generated, dtype=float)
    mean_gap = real.mean(axis=0) - generated.mean(axis=0)
    covariance_real = np.cov(real, rowvar=False)
    covariance_generated = np.cov(generated, rowvar=False)
    covariance_product_root = sqrtm(covariance_real @ covariance_generated)
    covariance_product_root = np.real_if_close(covariance_product_root, tol=1000).real
    value = (
        mean_gap @ mean_gap
        + np.trace(covariance_real)
        + np.trace(covariance_generated)
        - 2.0 * np.trace(covariance_product_root)
    )
    return float(max(value, 0.0))


def rbf_mmd_squared(real: np.ndarray, generated: np.ndarray) -> float:
    """Biased non-negative RBF maximum mean discrepancy with median bandwidth."""

    real = np.asarray(real, dtype=float)
    generated = np.asarray(generated, dtype=float)
    combined = np.vstack([real, generated])
    distances = pdist(combined, metric="sqeuclidean")
    positive = distances[distances > 0]
    bandwidth_sq = float(np.median(positive)) if len(positive) else 1.0
    bandwidth_sq = max(bandwidth_sq, 1e-12)
    kernel_rr = np.exp(-cdist(real, real, "sqeuclidean") / (2.0 * bandwidth_sq))
    kernel_gg = np.exp(
        -cdist(generated, generated, "sqeuclidean") / (2.0 * bandwidth_sq)
    )
    kernel_rg = np.exp(
        -cdist(real, generated, "sqeuclidean") / (2.0 * bandwidth_sq)
    )
    value = kernel_rr.mean() + kernel_gg.mean() - 2.0 * kernel_rg.mean()
    return float(max(value, 0.0))


def manifold_precision_recall(
    real: np.ndarray, generated: np.ndarray, n_neighbors: int = 5
) -> tuple[float, float]:
    """k-NN manifold precision and recall in the declared feature space."""

    if n_neighbors < 1 or min(len(real), len(generated)) <= n_neighbors:
        raise ValueError("not enough rows for the requested neighborhood")
    real_model = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(real)
    generated_model = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(generated)
    real_radii = real_model.kneighbors(real)[0][:, -1]
    generated_radii = generated_model.kneighbors(generated)[0][:, -1]
    fake_to_real_distance, fake_to_real_index = real_model.kneighbors(
        generated, n_neighbors=1
    )
    real_to_fake_distance, real_to_fake_index = generated_model.kneighbors(
        real, n_neighbors=1
    )
    precision = np.mean(
        fake_to_real_distance[:, 0] <= real_radii[fake_to_real_index[:, 0]]
    )
    recall = np.mean(
        real_to_fake_distance[:, 0] <= generated_radii[real_to_fake_index[:, 0]]
    )
    return float(precision), float(recall)


def label_diversity(
    train_latent: np.ndarray,
    train_labels: np.ndarray,
    generated: np.ndarray,
    n_neighbors: int = 5,
) -> tuple[float, int]:
    """Normalized entropy and class coverage under a fixed k-NN digit probe."""

    classifier = KNeighborsClassifier(n_neighbors=n_neighbors).fit(
        train_latent, train_labels
    )
    predicted = classifier.predict(generated)
    counts = np.bincount(predicted, minlength=10).astype(float)
    probabilities = counts / counts.sum()
    nonzero = probabilities > 0
    entropy = -float(np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))
    return entropy / np.log(10.0), int(nonzero.sum())


def nearest_train_distance(train: np.ndarray, generated: np.ndarray) -> float:
    distances = NearestNeighbors(n_neighbors=1).fit(train).kneighbors(generated)[0]
    return float(np.median(distances[:, 0]))


def evaluate_generator(
    real_test: np.ndarray,
    train_latent: np.ndarray,
    train_labels: np.ndarray,
    generated: np.ndarray,
) -> dict[str, float]:
    """Return complementary fidelity, coverage, diversity, and copying diagnostics."""

    precision, recall = manifold_precision_recall(real_test, generated)
    entropy, class_coverage = label_diversity(train_latent, train_labels, generated)
    return {
        "frechet_latent": frechet_gaussian_distance(real_test, generated),
        "mmd_rbf": rbf_mmd_squared(real_test, generated),
        "precision_knn": precision,
        "recall_knn": recall,
        "label_entropy": entropy,
        "class_coverage": float(class_coverage),
        "median_nearest_train": nearest_train_distance(train_latent, generated),
    }
