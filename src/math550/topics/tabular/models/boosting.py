"""Second-order boosting implementations inspired by XGBoost and LightGBM.

These classes intentionally implement the central algorithms rather than every
production feature. ``ScratchXGBoostClassifier`` grows exact depth-wise Newton
trees. ``ScratchLightGBMClassifier`` uses quantile histograms and chooses the
best leaf globally, reproducing LightGBM's two most important design choices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from math550.core.numerics import logit, sigmoid

from ..validation import check_binary_inputs, check_predictors


@dataclass
class _BoostNode:
    value: float
    depth: int
    feature: int | None = None
    threshold: float | None = None
    left: "_BoostNode | None" = None
    right: "_BoostNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


def _gain(
    gradient_left: float,
    hessian_left: float,
    gradient_right: float,
    hessian_right: float,
    reg_lambda: float,
    gamma: float,
) -> float:
    gradient_total = gradient_left + gradient_right
    hessian_total = hessian_left + hessian_right
    return 0.5 * (
        gradient_left**2 / (hessian_left + reg_lambda)
        + gradient_right**2 / (hessian_right + reg_lambda)
        - gradient_total**2 / (hessian_total + reg_lambda)
    ) - gamma


class _ExactNewtonTree:
    """Exact greedy, depth-wise tree for second-order objectives."""

    def __init__(
        self,
        max_depth: int,
        min_samples_leaf: int,
        min_child_weight: float,
        reg_lambda: float,
        gamma: float,
        feature_ids: np.ndarray,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.feature_ids = feature_ids

    def _leaf_value(self, indices: np.ndarray) -> float:
        return -float(self.gradient[indices].sum()) / (
            float(self.hessian[indices].sum()) + self.reg_lambda
        )

    def _best_split(self, indices: np.ndarray) -> tuple[int, float, float] | None:
        n_node = len(indices)
        total_gradient = float(self.gradient[indices].sum())
        total_hessian = float(self.hessian[indices].sum())
        best_gain = 0.0
        best: tuple[int, float, float] | None = None
        for feature in self.feature_ids:
            values = self.X[indices, feature]
            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]
            sorted_gradient = self.gradient[indices][order]
            sorted_hessian = self.hessian[indices][order]
            cumulative_gradient = np.cumsum(sorted_gradient)
            cumulative_hessian = np.cumsum(sorted_hessian)
            positions = np.arange(1, n_node)
            valid = (
                (sorted_values[:-1] < sorted_values[1:])
                & (positions >= self.min_samples_leaf)
                & ((n_node - positions) >= self.min_samples_leaf)
            )
            candidate_positions = positions[valid]
            if candidate_positions.size == 0:
                continue
            left_gradient = cumulative_gradient[candidate_positions - 1]
            left_hessian = cumulative_hessian[candidate_positions - 1]
            right_hessian = total_hessian - left_hessian
            curvature_ok = (
                np.minimum(left_hessian, right_hessian) >= self.min_child_weight
            )
            if not np.any(curvature_ok):
                continue
            candidate_positions = candidate_positions[curvature_ok]
            left_gradient = left_gradient[curvature_ok]
            left_hessian = left_hessian[curvature_ok]
            right_hessian = right_hessian[curvature_ok]
            right_gradient = total_gradient - left_gradient
            gains = 0.5 * (
                left_gradient**2 / (left_hessian + self.reg_lambda)
                + right_gradient**2 / (right_hessian + self.reg_lambda)
                - total_gradient**2 / (total_hessian + self.reg_lambda)
            ) - self.gamma
            local_index = int(np.argmax(gains))
            gain = float(gains[local_index])
            if gain > best_gain + 1e-14:
                position = int(candidate_positions[local_index])
                threshold = 0.5 * (
                    sorted_values[position - 1] + sorted_values[position]
                )
                best_gain = gain
                best = (int(feature), float(threshold), gain)
        return best

    def _grow(self, indices: np.ndarray, depth: int) -> _BoostNode:
        node = _BoostNode(value=self._leaf_value(indices), depth=depth)
        if depth >= self.max_depth or len(indices) < 2 * self.min_samples_leaf:
            return node
        split = self._best_split(indices)
        if split is None:
            return node
        feature, threshold, _ = split
        left_mask = self.X[indices, feature] <= threshold
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
        gradient: np.ndarray,
        hessian: np.ndarray,
        indices: np.ndarray,
    ) -> "_ExactNewtonTree":
        self.X = X
        self.gradient = gradient
        self.hessian = hessian
        self.root_ = self._grow(indices, depth=0)
        del self.X, self.gradient, self.hessian
        return self

    @staticmethod
    def _predict_one(row: np.ndarray, root: _BoostNode) -> float:
        node = root
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(row, self.root_) for row in X])


class ScratchXGBoostClassifier(ClassifierMixin, BaseEstimator):
    """Regularized Newton gradient boosting with exact depth-wise trees."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 2,
        min_samples_leaf: int = 2,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchXGBoostClassifier":
        X_array, y_array = check_binary_inputs(X, y)
        if self.n_estimators < 1 or self.learning_rate <= 0:
            raise ValueError("n_estimators and learning_rate must be positive")
        if not 0 < self.subsample <= 1 or not 0 < self.colsample_bytree <= 1:
            raise ValueError("subsample and colsample_bytree must lie in (0, 1]")
        self.n_features_in_ = X_array.shape[1]
        self.classes_ = np.array([0, 1])
        self.initial_score_ = logit(float(y_array.mean()))
        raw_score = np.full(len(y_array), self.initial_score_)
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.training_log_loss_ = []

        for _ in range(self.n_estimators):
            probability = sigmoid(raw_score)
            gradient = probability - y_array
            hessian = np.maximum(probability * (1.0 - probability), 1e-12)
            row_count = max(2, int(np.ceil(self.subsample * len(y_array))))
            row_ids = (
                rng.choice(len(y_array), row_count, replace=False)
                if row_count < len(y_array)
                else np.arange(len(y_array))
            )
            feature_count = max(1, int(np.ceil(self.colsample_bytree * self.n_features_in_)))
            feature_ids = rng.choice(self.n_features_in_, feature_count, replace=False)
            tree = _ExactNewtonTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                feature_ids=feature_ids,
            ).fit(X_array, gradient, hessian, row_ids)
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


class _HistogramLeafWiseTree:
    """Histogram Newton tree grown by globally best leaf gain."""

    def __init__(
        self,
        num_leaves: int,
        max_depth: int,
        min_samples_leaf: int,
        min_child_weight: float,
        reg_lambda: float,
        gamma: float,
        feature_ids: np.ndarray,
        thresholds: list[np.ndarray],
    ) -> None:
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.feature_ids = feature_ids
        self.thresholds = thresholds

    def _leaf_value(self, indices: np.ndarray) -> float:
        return -float(self.gradient[indices].sum()) / (
            float(self.hessian[indices].sum()) + self.reg_lambda
        )

    def _best_histogram_split(
        self, indices: np.ndarray
    ) -> tuple[int, float, float, np.ndarray, np.ndarray] | None:
        total_gradient = float(self.gradient[indices].sum())
        total_hessian = float(self.hessian[indices].sum())
        best: tuple[int, float, float, np.ndarray, np.ndarray] | None = None
        best_gain = 0.0
        for feature in self.feature_ids:
            feature_thresholds = self.thresholds[feature]
            if len(feature_thresholds) == 0:
                continue
            bins = np.searchsorted(
                feature_thresholds, self.X[indices, feature], side="right"
            )
            n_bins = len(feature_thresholds) + 1
            gradient_hist = np.bincount(
                bins, weights=self.gradient[indices], minlength=n_bins
            )
            hessian_hist = np.bincount(
                bins, weights=self.hessian[indices], minlength=n_bins
            )
            count_hist = np.bincount(bins, minlength=n_bins)
            cumulative_gradient = np.cumsum(gradient_hist)[:-1]
            cumulative_hessian = np.cumsum(hessian_hist)[:-1]
            cumulative_count = np.cumsum(count_hist)[:-1]
            for bin_id in range(len(feature_thresholds)):
                left_count = int(cumulative_count[bin_id])
                right_count = len(indices) - left_count
                if min(left_count, right_count) < self.min_samples_leaf:
                    continue
                gradient_left = float(cumulative_gradient[bin_id])
                hessian_left = float(cumulative_hessian[bin_id])
                hessian_right = total_hessian - hessian_left
                if min(hessian_left, hessian_right) < self.min_child_weight:
                    continue
                gain = _gain(
                    gradient_left,
                    hessian_left,
                    total_gradient - gradient_left,
                    hessian_right,
                    self.reg_lambda,
                    self.gamma,
                )
                if gain > best_gain + 1e-14:
                    threshold = float(feature_thresholds[bin_id])
                    left_mask = self.X[indices, feature] <= threshold
                    best_gain = gain
                    best = (
                        int(feature),
                        threshold,
                        float(gain),
                        indices[left_mask],
                        indices[~left_mask],
                    )
        return best

    def fit(
        self,
        X: np.ndarray,
        gradient: np.ndarray,
        hessian: np.ndarray,
        indices: np.ndarray,
    ) -> "_HistogramLeafWiseTree":
        self.X = X
        self.gradient = gradient
        self.hessian = hessian
        self.root_ = _BoostNode(value=self._leaf_value(indices), depth=0)
        leaves: list[tuple[_BoostNode, np.ndarray]] = [(self.root_, indices)]

        while len(leaves) < self.num_leaves:
            candidate_index = None
            candidate_split = None
            candidate_gain = 0.0
            for leaf_index, (node, leaf_indices) in enumerate(leaves):
                if node.depth >= self.max_depth:
                    continue
                split = self._best_histogram_split(leaf_indices)
                if split is not None and split[2] > candidate_gain:
                    candidate_index = leaf_index
                    candidate_split = split
                    candidate_gain = split[2]
            if candidate_index is None or candidate_split is None:
                break
            node, _ = leaves.pop(candidate_index)
            feature, threshold, _, left_indices, right_indices = candidate_split
            node.feature = feature
            node.threshold = threshold
            node.left = _BoostNode(
                value=self._leaf_value(left_indices), depth=node.depth + 1
            )
            node.right = _BoostNode(
                value=self._leaf_value(right_indices), depth=node.depth + 1
            )
            leaves.append((node.left, left_indices))
            leaves.append((node.right, right_indices))
        del self.X, self.gradient, self.hessian
        return self

    @staticmethod
    def _predict_one(row: np.ndarray, root: _BoostNode) -> float:
        node = root
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(row, self.root_) for row in X])


class ScratchLightGBMClassifier(ClassifierMixin, BaseEstimator):
    """Histogram-based, leaf-wise Newton boosting for binary classification."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        num_leaves: int = 7,
        max_depth: int = 4,
        max_bins: int = 32,
        min_samples_leaf: int = 10,
        min_child_weight: float = 1e-3,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    def _make_thresholds(self, X: np.ndarray) -> list[np.ndarray]:
        quantiles = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
        thresholds = []
        for feature in range(X.shape[1]):
            values = np.unique(np.quantile(X[:, feature], quantiles))
            minimum, maximum = X[:, feature].min(), X[:, feature].max()
            thresholds.append(values[(values > minimum) & (values < maximum)])
        return thresholds

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchLightGBMClassifier":
        X_array, y_array = check_binary_inputs(X, y)
        if self.n_estimators < 1 or self.learning_rate <= 0 or self.max_bins < 2:
            raise ValueError("n_estimators, learning_rate, and max_bins must be positive")
        if self.num_leaves < 2 or self.max_depth < 1:
            raise ValueError("num_leaves >= 2 and max_depth >= 1 are required")
        if not 0 < self.subsample <= 1 or not 0 < self.colsample_bytree <= 1:
            raise ValueError("subsample and colsample_bytree must lie in (0, 1]")
        self.n_features_in_ = X_array.shape[1]
        self.classes_ = np.array([0, 1])
        self.initial_score_ = logit(float(y_array.mean()))
        self.bin_thresholds_ = self._make_thresholds(X_array)
        raw_score = np.full(len(y_array), self.initial_score_)
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.training_log_loss_ = []

        for _ in range(self.n_estimators):
            probability = sigmoid(raw_score)
            gradient = probability - y_array
            hessian = np.maximum(probability * (1.0 - probability), 1e-12)
            row_count = max(2, int(np.ceil(self.subsample * len(y_array))))
            row_ids = (
                rng.choice(len(y_array), row_count, replace=False)
                if row_count < len(y_array)
                else np.arange(len(y_array))
            )
            feature_count = max(1, int(np.ceil(self.colsample_bytree * self.n_features_in_)))
            feature_ids = rng.choice(self.n_features_in_, feature_count, replace=False)
            tree = _HistogramLeafWiseTree(
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                feature_ids=feature_ids,
                thresholds=self.bin_thresholds_,
            ).fit(X_array, gradient, hessian, row_ids)
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
