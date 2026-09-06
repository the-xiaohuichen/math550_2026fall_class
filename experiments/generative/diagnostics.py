"""Paired uncertainty and exactness checks for the generative benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from math550.topics.generative import GaussianAffineFlow


def paired_bootstrap_summary(
    metrics: pd.DataFrame, n_bootstrap: int, random_state: int
) -> pd.DataFrame:
    """Summarize MLP-minus-linear gaps by paired sample-generation seed."""

    metric_names = [
        "frechet_latent",
        "mmd_rbf",
        "precision_knn",
        "recall_knn",
        "label_entropy",
        "class_coverage",
        "median_nearest_train",
    ]
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_state)
    for family in ["Flow matching", "Diffusion"]:
        family_table = metrics[metrics["family"] == family]
        pivot = family_table.pivot(index="replicate", columns="implementation")
        for metric in metric_names:
            differences = (
                pivot[(metric, "mlp")].to_numpy()
                - pivot[(metric, "linear ablation")].to_numpy()
            )
            indices = rng.integers(
                0, len(differences), size=(n_bootstrap, len(differences))
            )
            bootstrap = differences[indices].mean(axis=1)
            rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "mlp_mean": float(pivot[(metric, "mlp")].mean()),
                    "linear_mean": float(
                        pivot[(metric, "linear ablation")].mean()
                    ),
                    "gap": float(differences.mean()),
                    "gap_ci_low": float(np.quantile(bootstrap, 0.025)),
                    "gap_ci_high": float(np.quantile(bootstrap, 0.975)),
                    "replicates": len(differences),
                }
            )
    return pd.DataFrame(rows)


def affine_flow_exactness(train: np.ndarray, test: np.ndarray) -> pd.DataFrame:
    """Compare change-of-variables code with SciPy's multivariate Gaussian."""

    flow = GaussianAffineFlow(jitter=1e-6).fit(train)
    covariance = flow.cholesky_ @ flow.cholesky_.T
    scratch = flow.score_samples(test)
    library = multivariate_normal(mean=flow.mean_, cov=covariance).logpdf(test)
    latent, inverse_log_det = flow.transform(test)
    reconstructed, forward_log_det = flow.inverse_transform(latent)
    return pd.DataFrame(
        [
            {
                "test_rows": len(test),
                "max_log_density_abs_gap": float(np.max(np.abs(scratch - library))),
                "mean_log_density_abs_gap": float(np.mean(np.abs(scratch - library))),
                "max_round_trip_abs_gap": float(np.max(np.abs(test - reconstructed))),
                "max_logdet_cancellation_gap": float(
                    np.max(np.abs(inverse_log_det + forward_log_det))
                ),
            }
        ]
    )
