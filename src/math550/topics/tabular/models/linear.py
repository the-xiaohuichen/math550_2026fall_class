"""From-scratch linear classifiers.

The implementation below exposes the mathematical pieces students usually do
not see in a library call: binary cross-entropy, its gradient, a Lipschitz step
size, and the proximal operator used for L1 regularization.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from math550.core.numerics import sigmoid, soft_threshold

from ..validation import check_binary_inputs, check_predictors


class ScratchLogisticRegression(ClassifierMixin, BaseEstimator):
    """Binary logistic regression trained by accelerated proximal gradient.

    The minimized objective is

    ``mean(log(1 + exp(x_i^T w)) - y_i x_i^T w) + penalty(w)``.

    The intercept is never penalized. For L2, ``penalty = lambda/2 ||w||^2``;
    for L1, ``penalty = lambda ||w||_1``. Setting ``regularization='none'``
    gives the maximum-likelihood objective. FISTA acceleration is used with the
    exact global Lipschitz bound of the smooth logistic objective.
    """

    def __init__(
        self,
        regularization: str = "none",
        reg_strength: float = 0.0,
        max_iter: int = 5_000,
        tol: float = 1e-8,
        fit_intercept: bool = True,
        step_size: float | None = None,
        accelerated: bool = True,
    ) -> None:
        self.regularization = regularization
        self.reg_strength = reg_strength
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.step_size = step_size
        self.accelerated = accelerated

    def _validate_hyperparameters(self) -> None:
        if self.regularization not in {"none", "l1", "l2"}:
            raise ValueError("regularization must be 'none', 'l1', or 'l2'")
        if self.reg_strength < 0:
            raise ValueError("reg_strength must be non-negative")
        if self.max_iter < 1 or self.tol <= 0:
            raise ValueError("max_iter and tol must be positive")
        if self.step_size is not None and self.step_size <= 0:
            raise ValueError("step_size must be positive")

    def _augment(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        return np.column_stack([np.ones(len(X)), X])

    def _penalized_slice(self, weights: np.ndarray) -> slice:
        return slice(1, None) if self.fit_intercept else slice(None)

    def _objective_components(
        self, X: np.ndarray, y: np.ndarray, weights: np.ndarray
    ) -> tuple[float, float]:
        """Return the data-fit and regularization terms separately."""

        scores = X @ weights
        data_loss = float(np.mean(np.logaddexp(0.0, scores) - y * scores))
        penalized = weights[self._penalized_slice(weights)]
        regularization_loss = 0.0
        if self.regularization == "l1":
            regularization_loss = self.reg_strength * float(np.abs(penalized).sum())
        elif self.regularization == "l2":
            regularization_loss = 0.5 * self.reg_strength * float(
                penalized @ penalized
            )
        return data_loss, regularization_loss

    def _objective(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        data_loss, regularization_loss = self._objective_components(X, y, weights)
        return data_loss + regularization_loss

    def _smooth_gradient(
        self, X: np.ndarray, y: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        gradient = X.T @ (sigmoid(X @ weights) - y) / len(y)
        if self.regularization == "l2":
            penalized = self._penalized_slice(weights)
            gradient[penalized] += self.reg_strength * weights[penalized]
        return gradient

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchLogisticRegression":
        self._validate_hyperparameters()
        X_array, y_array = check_binary_inputs(X, y)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X_array.shape[1]
        design = self._augment(X_array)

        spectral_norm = float(np.linalg.norm(design, ord=2))
        l2_curvature = self.reg_strength if self.regularization == "l2" else 0.0
        lipschitz = 0.25 * spectral_norm**2 / len(design) + l2_curvature
        step = self.step_size if self.step_size is not None else 1.0 / max(lipschitz, 1e-12)

        # Standard initialization
        weights = np.zeros(design.shape[1], dtype=float)
        extrapolated = weights.copy()
        momentum = 1.0
        initial_data_loss, initial_regularization_loss = self._objective_components(
            design, y_array, weights
        )
        self.data_loss_curve_ = [initial_data_loss]
        self.regularization_loss_curve_ = [initial_regularization_loss]
        self.loss_curve_ = [initial_data_loss + initial_regularization_loss]
        self.converged_ = False

        for iteration in range(1, self.max_iter + 1):
            # Forward step evaluated at the extrapolated point
            candidate = extrapolated - step * self._smooth_gradient(
                design, y_array, extrapolated
            )
            # Proximal (i.e., backward) step for L1 regularization
            if self.regularization == "l1":
                penalized = self._penalized_slice(candidate)
                candidate[penalized] = soft_threshold(
                    candidate[penalized], step * self.reg_strength
                )

            candidate_data_loss, candidate_regularization_loss = (
                self._objective_components(design, y_array, candidate)
            )
            candidate_objective = candidate_data_loss + candidate_regularization_loss

            # FISTA can overshoot. Restart from the accepted iterate and, if
            # necessary, backtrack until the penalized objective decreases.
            if candidate_objective > self.loss_curve_[-1] + 1e-12:
                # This is the case where momentum has produced a candidate increasing the objective function value, then restart at previous weight.
                extrapolated = weights.copy()
                momentum = 1.0
                # Backtracking to halve the step size immediately
                while candidate_objective > self.loss_curve_[-1] + 1e-12:
                    step *= 0.5
                    candidate = weights - step * self._smooth_gradient(
                        design, y_array, weights
                    )
                    if self.regularization == "l1":
                        penalized = self._penalized_slice(candidate)
                        candidate[penalized] = soft_threshold(
                            candidate[penalized], step * self.reg_strength
                        )
                    candidate_data_loss, candidate_regularization_loss = (
                        self._objective_components(design, y_array, candidate)
                    )
                    candidate_objective = (
                        candidate_data_loss + candidate_regularization_loss
                    )

            difference = np.linalg.norm(candidate - weights)
            scale = 1.0 + np.linalg.norm(weights)
            self.data_loss_curve_.append(candidate_data_loss)
            self.regularization_loss_curve_.append(candidate_regularization_loss)
            self.loss_curve_.append(candidate_objective)
            if difference <= self.tol * scale:
                weights = candidate
                self.n_iter_ = iteration
                self.converged_ = True
                break

            if self.accelerated:
                # FISTA momentum recurrence
                next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
                extrapolated = candidate + (
                    (momentum - 1.0) / next_momentum
                ) * (candidate - weights)
                momentum = next_momentum
            else:
                extrapolated = candidate
            weights = candidate
        else:
            self.n_iter_ = self.max_iter

        if self.fit_intercept:
            self.intercept_ = np.array([weights[0]])
            self.coef_ = weights[1:].reshape(1, -1)
        else:
            self.intercept_ = np.array([0.0])
            self.coef_ = weights.reshape(1, -1)
        self.final_loss_ = float(self.loss_curve_[-1])
        self.final_data_loss_ = float(self.data_loss_curve_[-1])
        self.final_regularization_loss_ = float(
            self.regularization_loss_curve_[-1]
        )
        self.step_size_ = float(step)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_array = check_predictors(X, self.n_features_in_)
        return X_array @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def cross_entropy_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return unpenalized mean cross-entropy on supplied observations."""

        X_array = check_predictors(X, self.n_features_in_)
        y_array = np.asarray(y, dtype=float).reshape(-1)
        scores = self.decision_function(X_array)
        return float(np.mean(np.logaddexp(0.0, scores) - y_array * scores))

    def objective_components(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate data loss, penalty, and total objective at fitted parameters."""

        X_array = check_predictors(X, self.n_features_in_)
        y_array = np.asarray(y, dtype=float).reshape(-1)
        if len(y_array) != len(X_array) or not np.isin(y_array, [0.0, 1.0]).all():
            raise ValueError("y must be a binary vector aligned with X")
        weights = (
            np.r_[self.intercept_[0], self.coef_[0]]
            if self.fit_intercept
            else self.coef_[0].copy()
        )
        data_loss, regularization_loss = self._objective_components(
            self._augment(X_array), y_array, weights
        )
        return {
            "data_loss": data_loss,
            "regularization_loss": regularization_loss,
            "objective": data_loss + regularization_loss,
        }
