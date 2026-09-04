"""Compact CART-style trees implemented with NumPy primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from ..validation import check_binary_inputs, check_predictors


@dataclass
class _TreeNode:
    value: float                        # Weighted class proportions
    feature: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


def _feature_count(max_features: str | int | float | None, n_features: int) -> int:
    if max_features is None:
        return n_features
    if max_features == "sqrt":
        return max(1, int(np.sqrt(n_features)))
    if max_features == "log2":
        return max(1, int(np.log2(n_features)))
    if isinstance(max_features, int):
        return min(n_features, max(1, max_features))
    if isinstance(max_features, float):
        return min(n_features, max(1, int(np.ceil(max_features * n_features))))
    raise ValueError("Unsupported max_features value")


class ScratchDecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    """Binary CART tree using weighted Gini impurity.

    Sample weights make the same primitive reusable as an AdaBoost weak learner.
    Thresholds are exact midpoints between distinct sorted training values.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | float | None = None,
        min_impurity_decrease: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

    @staticmethod
    def _gini(positive_weight: float, total_weight: float) -> float:
        if total_weight <= 0:
            return 0.0
        probability = positive_weight / total_weight
        return 2.0 * probability * (1.0 - probability)

    # Output of best split is a tuple: (split feature, threshold, gain)
    def _best_split(
        self, indices: np.ndarray, parent_impurity: float
    ) -> tuple[int, float, float] | None:
        n_node = len(indices)
        weights = self._sample_weight[indices]
        targets = self._y[indices]
        total_weight = float(weights.sum())
        total_positive = float(weights @ targets)
        # Randomly sample a subset of features to split with a given rng seed and then compare the gain
        feature_ids = self._rng.choice(
            self.n_features_in_, self._n_candidate_features, replace=False
        )
        best: tuple[int, float, float] | None = None
        best_gain = self.min_impurity_decrease

        for feature in feature_ids:
            values = self._X[indices, feature]
            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]
            sorted_weights = weights[order]
            sorted_targets = targets[order]
            cumulative_weight = np.cumsum(sorted_weights)
            cumulative_positive = np.cumsum(sorted_weights * sorted_targets)
            positions = np.arange(1, n_node)
            valid = (
                (sorted_values[:-1] < sorted_values[1:])
                & (positions >= self.min_samples_leaf)
                & ((n_node - positions) >= self.min_samples_leaf)
            )
            for position in positions[valid]:
                left_weight = float(cumulative_weight[position - 1])
                right_weight = total_weight - left_weight
                left_positive = float(cumulative_positive[position - 1])
                right_positive = total_positive - left_positive
                child_impurity = (
                    left_weight * self._gini(left_positive, left_weight)
                    + right_weight * self._gini(right_positive, right_weight)
                ) / total_weight
                gain = parent_impurity - child_impurity
                if gain > best_gain + 1e-14:
                    # Mid-point splitting rule
                    threshold = 0.5 * (
                        sorted_values[position - 1] + sorted_values[position]
                    )
                    best_gain = gain
                    best = (int(feature), float(threshold), float(gain))
        return best

    def _grow(self, indices: np.ndarray, depth: int) -> _TreeNode:
        weights = self._sample_weight[indices]
        targets = self._y[indices]
        total_weight = float(weights.sum())
        probability = float(weights @ targets / total_weight)
        node = _TreeNode(value=probability)
        impurity = self._gini(float(weights @ targets), total_weight)
        stop = (
            impurity <= 1e-15
            or len(indices) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        )
        if stop:
            return node

        split = self._best_split(indices, impurity)
        if split is None:
            return node
        feature, threshold, _ = split
        left_mask = self._X[indices, feature] <= threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]
        if min(len(left_indices), len(right_indices)) < self.min_samples_leaf:
            return node
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(left_indices, depth + 1)
        node.right = self._grow(right_indices, depth + 1)
        return node

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "ScratchDecisionTreeClassifier":
        X_array, y_array = check_binary_inputs(X, y)
        if self.min_samples_split < 2 or self.min_samples_leaf < 1:
            raise ValueError("min_samples_split >= 2 and min_samples_leaf >= 1 are required")
        self._X = X_array
        self._y = y_array
        self.n_features_in_ = X_array.shape[1]
        self.classes_ = np.array([0, 1])
        if sample_weight is None:
            self._sample_weight = np.ones(len(y_array), dtype=float)
        else:
            self._sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
            if len(self._sample_weight) != len(y_array) or np.any(self._sample_weight < 0):
                raise ValueError("sample_weight must be non-negative with one value per row")
        if self._sample_weight.sum() <= 0:
            raise ValueError("sample_weight must have positive total weight")

        # This determines whether the seed is fixed (single decision tree) or random (random forests for random candidate feature subsets)
        self._rng = np.random.default_rng(self.random_state)
        self._n_candidate_features = _feature_count(self.max_features, self.n_features_in_)
        self.root_ = self._grow(np.arange(len(y_array)), depth=0)
        del self._X, self._y, self._sample_weight
        return self

    def _predict_one(self, row: np.ndarray) -> float:
        node = self.root_
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_array = check_predictors(X, self.n_features_in_)
        positive = np.array([self._predict_one(row) for row in X_array])
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ScratchRegressionTree(RegressorMixin, BaseEstimator):
    """CART regression tree using exact squared-error splits."""

    def __init__(
        self,
        max_depth: int | None = 2,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | float | None = None,
        min_impurity_decrease: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

    @staticmethod
    def _sse(total: float, squared_total: float, count: int) -> float:
        if count == 0:
            return 0.0
        return max(0.0, squared_total - total**2 / count)

    def _best_split(
        self, indices: np.ndarray, parent_sse: float
    ) -> tuple[int, float, float] | None:
        n_node = len(indices)
        targets = self._y[indices]
        total = float(targets.sum())
        squared_total = float(targets @ targets)
        features = self._rng.choice(
            self.n_features_in_, self._n_candidate_features, replace=False
        )
        best: tuple[int, float, float] | None = None
        best_gain = self.min_impurity_decrease
        for feature in features:
            values = self._X[indices, feature]
            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]
            sorted_targets = targets[order]
            cumulative = np.cumsum(sorted_targets)
            cumulative_squared = np.cumsum(sorted_targets**2)
            positions = np.arange(1, n_node)
            valid = (
                (sorted_values[:-1] < sorted_values[1:])
                & (positions >= self.min_samples_leaf)
                & ((n_node - positions) >= self.min_samples_leaf)
            )
            for position in positions[valid]:
                left_total = float(cumulative[position - 1])
                left_squared = float(cumulative_squared[position - 1])
                right_total = total - left_total
                right_squared = squared_total - left_squared
                children_sse = self._sse(left_total, left_squared, int(position)) + self._sse(
                    right_total, right_squared, n_node - int(position)
                )
                gain = parent_sse - children_sse
                if gain > best_gain + 1e-14:
                    threshold = 0.5 * (
                        sorted_values[position - 1] + sorted_values[position]
                    )
                    best_gain = gain
                    best = (int(feature), float(threshold), float(gain))
        return best

    def _grow(self, indices: np.ndarray, depth: int) -> _TreeNode:
        targets = self._y[indices]
        node = _TreeNode(value=float(targets.mean()))
        parent_sse = float(((targets - targets.mean()) ** 2).sum())
        stop = (
            parent_sse <= 1e-15
            or len(indices) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        )
        if stop:
            return node
        split = self._best_split(indices, parent_sse)
        if split is None:
            return node
        feature, threshold, _ = split
        left_mask = self._X[indices, feature] <= threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]
        if min(len(left_indices), len(right_indices)) < self.min_samples_leaf:
            return node
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(left_indices, depth + 1)
        node.right = self._grow(right_indices, depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchRegressionTree":
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y, dtype=float).reshape(-1)
        if X_array.ndim != 2 or len(X_array) != len(y_array):
            raise ValueError("X must be 2D and aligned with y")
        if not np.isfinite(X_array).all() or not np.isfinite(y_array).all():
            raise ValueError("X and y must be finite")
        self._X = X_array
        self._y = y_array
        self.n_features_in_ = X_array.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self._n_candidate_features = _feature_count(self.max_features, self.n_features_in_)
        self.root_ = self._grow(np.arange(len(y_array)), depth=0)
        del self._X, self._y
        return self

    def _predict_one(self, row: np.ndarray) -> float:
        node = self.root_
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_array = check_predictors(X, self.n_features_in_)
        return np.array([self._predict_one(row) for row in X_array])
