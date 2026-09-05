"""Bagging and first-order boosting built from the scratch tree primitives."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from math550.core.numerics import logit, sigmoid

from ..validation import check_binary_inputs, check_predictors
from .trees import ScratchDecisionTreeClassifier, ScratchRegressionTree


class ScratchRandomForestClassifier(ClassifierMixin, BaseEstimator):
    """Bootstrap aggregation of randomized scratch CART trees."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 5,
        min_samples_leaf: int = 1,
        max_features: str | int | float | None = "sqrt",
        bootstrap: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchRandomForestClassifier":
        X_array, y_array = check_binary_inputs(X, y)
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        self.n_features_in_ = X_array.shape[1]
        self.classes_ = np.array([0, 1])
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.bootstrap_indices_ = []
        for _ in range(self.n_estimators):
            # Bagging (randomly sampling data with replacement)
            indices = (
                rng.integers(0, len(y_array), size=len(y_array))
                if self.bootstrap
                else np.arange(len(y_array))
            )
            # Generate another random seed for each tree
            tree = ScratchDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
            tree.fit(X_array[indices], y_array[indices])
            self.estimators_.append(tree)
            self.bootstrap_indices_.append(indices)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_array = check_predictors(X, self.n_features_in_)
        positive = np.mean(
            [tree.predict_proba(X_array)[:, 1] for tree in self.estimators_], axis=0
        )
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ScratchAdaBoostClassifier(ClassifierMixin, BaseEstimator):
    """Binary discrete AdaBoost using weighted scratch decision stumps."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchAdaBoostClassifier":
        X_array, y_array = check_binary_inputs(X, y)
        if self.n_estimators < 1 or self.learning_rate <= 0:
            raise ValueError("n_estimators and learning_rate must be positive")
        self.n_features_in_ = X_array.shape[1]
        self.classes_ = np.array([0, 1])
        signed_y = 2.0 * y_array - 1.0
        sample_weight = np.full(len(y_array), 1.0 / len(y_array))
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.estimator_weights_ = []
        self.training_exponential_loss_ = []

        for _ in range(self.n_estimators):
            stump = ScratchDecisionTreeClassifier(
                max_depth=1,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
            stump.fit(X_array, y_array, sample_weight=sample_weight)
            # Convert predicted labels from {0, 1} to {-1, 1}
            signed_prediction = 2.0 * stump.predict(X_array) - 1.0
            # Compute the adaptive error
            error = float(sample_weight[signed_prediction != signed_y].sum())
            if error >= 0.5 - 1e-14:
                break
            error = float(np.clip(error, 1e-12, 1.0 - 1e-12))
            # Weighted mis-classification error
            alpha = self.learning_rate * 0.5 * np.log((1.0 - error) / error)
            sample_weight *= np.exp(-alpha * signed_y * signed_prediction)
            sample_weight /= sample_weight.sum()
            self.estimators_.append(stump)
            self.estimator_weights_.append(float(alpha))
            margin = self.decision_function(X_array)
            self.training_exponential_loss_.append(float(np.mean(np.exp(-signed_y * margin))))
            if error <= 1e-12:
                break
        if not self.estimators_:
            raise RuntimeError("AdaBoost could not fit a weak learner better than chance")
        self.estimator_weights_ = np.asarray(self.estimator_weights_)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_array = check_predictors(X, self.n_features_in_)
        score = np.zeros(len(X_array), dtype=float)
        for alpha, stump in zip(self.estimator_weights_, self.estimators_):
            score += alpha * (2.0 * stump.predict(X_array) - 1.0)
        return score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = sigmoid(2.0 * self.decision_function(X))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0).astype(int)


class ScratchGradientBoostingClassifier(ClassifierMixin, BaseEstimator):
    """First-order gradient boosting for binomial cross-entropy.

    Each shallow regression tree approximates the negative functional gradient
    ``y - p``. This is the core idea behind classical gradient boosting; the
    second-order Newton version is implemented separately in ``boosting.py``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 2,
        min_samples_leaf: int = 5,
        subsample: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchGradientBoostingClassifier":
        X_array, y_array = check_binary_inputs(X, y)
        if self.n_estimators < 1 or self.learning_rate <= 0:
            raise ValueError("n_estimators and learning_rate must be positive")
        if not 0 < self.subsample <= 1:
            raise ValueError("subsample must lie in (0, 1]")
        self.n_features_in_ = X_array.shape[1]
        self.classes_ = np.array([0, 1])
        self.initial_score_ = logit(float(y_array.mean()))
        raw_score = np.full(len(y_array), self.initial_score_)
        self.estimators_ = []
        self.training_log_loss_ = []
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            probabilities = sigmoid(raw_score)
            negative_gradient = y_array - probabilities
            sample_size = max(2, int(np.ceil(self.subsample * len(y_array))))
            indices = (
                rng.choice(len(y_array), sample_size, replace=False)
                if sample_size < len(y_array)
                else np.arange(len(y_array))
            )
            tree = ScratchRegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
            tree.fit(X_array[indices], negative_gradient[indices])
            raw_score += self.learning_rate * tree.predict(X_array)
            self.estimators_.append(tree)
            self.training_log_loss_.append(
                float(np.mean(np.logaddexp(0.0, raw_score) - y_array * raw_score))
            )
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_array = check_predictors(X, self.n_features_in_)
        score = np.full(len(X_array), self.initial_score_)
        for tree in self.estimators_:
            score += self.learning_rate * tree.predict(X_array)
        return score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
