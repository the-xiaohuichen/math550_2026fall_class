"""Parameter, objective, and regularization diagnostics for tabular models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from math550.topics.tabular import ScratchLogisticRegression

from .config import PairSpec


def logistic_parameter_checks(
    fitted: dict[str, object], X_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:
    """Compare parameters and evaluate both fits under one objective convention."""

    rows = []
    for family in ["Logistic none", "Logistic L1", "Logistic L2"]:
        scratch_pipeline = fitted[f"{family}|scratch"]
        library_pipeline = fitted[f"{family}|library"]
        scratch = scratch_pipeline.named_steps["model"]
        library = library_pipeline.named_steps["model"]
        X_scaled = scratch_pipeline.named_steps["scale"].transform(X_train)
        scratch_coef = scratch.coef_[0]
        library_coef = library.coef_[0]
        scratch_components = scratch.objective_components(X_scaled, y_train)
        library_scores = X_scaled @ library_coef + library.intercept_[0]
        library_data_loss = float(
            np.mean(
                np.logaddexp(0.0, library_scores)
                - y_train.to_numpy() * library_scores
            )
        )
        if scratch.regularization == "l1":
            library_penalty = scratch.reg_strength * float(
                np.abs(library_coef).sum()
            )
        elif scratch.regularization == "l2":
            library_penalty = 0.5 * scratch.reg_strength * float(
                library_coef @ library_coef
            )
        else:
            library_penalty = 0.0
        library_objective = library_data_loss + library_penalty
        scratch_support = np.abs(scratch_coef) > 1e-8
        library_support = np.abs(library_coef) > 1e-8
        union = int(np.logical_or(scratch_support, library_support).sum())
        intersection = int(np.logical_and(scratch_support, library_support).sum())
        rows.append(
            {
                "family": family,
                "coefficient_correlation": np.corrcoef(
                    scratch_coef, library_coef
                )[0, 1],
                "coefficient_mae": np.mean(np.abs(scratch_coef - library_coef)),
                "intercept_abs_gap": abs(
                    scratch.intercept_[0] - library.intercept_[0]
                ),
                "scratch_nonzero": int(scratch_support.sum()),
                "library_nonzero": int(library_support.sum()),
                "support_jaccard": intersection / union if union else 1.0,
                "scratch_iterations": scratch.n_iter_,
                "scratch_converged": scratch.converged_,
                "scratch_data_loss": scratch_components["data_loss"],
                "scratch_regularization_loss": scratch_components[
                    "regularization_loss"
                ],
                "scratch_final_objective": scratch_components["objective"],
                "library_data_loss": library_data_loss,
                "library_regularization_loss": library_penalty,
                "library_objective": library_objective,
                "objective_abs_gap": abs(
                    scratch_components["objective"] - library_objective
                ),
            }
        )
    return pd.DataFrame(rows)


def model_configuration_table(pairs: list[PairSpec]) -> pd.DataFrame:
    """Return matched settings and intentional implementation differences."""

    return pd.DataFrame(
        [
            {
                "family": pair.family,
                "matched_parameters": pair.matched_parameters,
                "known_differences": pair.known_differences,
            }
            for pair in pairs
        ]
    )


def regularization_diagnostic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    """Evaluate L1/L2 strength on an inner split without touching test data."""

    X_inner, X_validation, y_inner, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        stratify=y_train,
        random_state=random_state + 1,
    )
    scaler = StandardScaler()
    X_inner_scaled = scaler.fit_transform(X_inner)
    X_validation_scaled = scaler.transform(X_validation)
    strengths = np.logspace(-4, -0.5, 9)
    rows = []
    for penalty in ["l1", "l2"]:
        for strength in strengths:
            model = ScratchLogisticRegression(
                regularization=penalty,
                reg_strength=float(strength),
                max_iter=6_000,
                tol=1e-8,
            ).fit(X_inner_scaled, y_inner)
            probability = model.predict_proba(X_validation_scaled)[:, 1]
            rows.append(
                {
                    "penalty": penalty.upper(),
                    "reg_strength": strength,
                    "validation_log_loss": log_loss(y_validation, probability),
                    "coefficient_l1_norm": np.abs(model.coef_[0]).sum(),
                    "coefficient_l2_norm": np.linalg.norm(model.coef_[0]),
                    "nonzero_coefficients": int(
                        (np.abs(model.coef_[0]) > 1e-8).sum()
                    ),
                    "iterations": model.n_iter_,
                }
            )
    return pd.DataFrame(rows)
