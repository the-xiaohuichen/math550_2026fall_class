"""Invertible transformations used to teach normalizing-flow mechanics."""

from __future__ import annotations

import numpy as np


class GaussianAffineFlow:
    """Exact affine normalizing flow fitted by the Gaussian maximum likelihood estimate."""

    def __init__(self, jitter: float = 1e-6) -> None:
        if jitter <= 0:
            raise ValueError("jitter must be positive")
        self.jitter = float(jitter)

    def fit(self, X: np.ndarray) -> "GaussianAffineFlow":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < 2:
            raise ValueError("X must be a matrix with at least two rows")
        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_
        covariance = centered.T @ centered / len(X)
        covariance += self.jitter * np.eye(X.shape[1])
        self.cholesky_ = np.linalg.cholesky(covariance)
        self.log_abs_det_ = float(np.log(np.diag(self.cholesky_)).sum())
        self.dimension_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map data to base noise and return log |det dz/dx| per row."""

        X = np.asarray(X, dtype=float)
        latent = np.linalg.solve(self.cholesky_, (X - self.mean_).T).T
        return latent, np.full(len(X), -self.log_abs_det_)

    def inverse_transform(self, latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map base noise to data and return log |det dx/dz| per row."""

        latent = np.asarray(latent, dtype=float)
        data = latent @ self.cholesky_.T + self.mean_
        return data, np.full(len(latent), self.log_abs_det_)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        latent, inverse_log_det = self.transform(X)
        base_log_density = -0.5 * (
            self.dimension_ * np.log(2.0 * np.pi) + np.sum(latent**2, axis=1)
        )
        return base_log_density + inverse_log_det

    def sample(self, n_samples: int, random_state: int = 550) -> np.ndarray:
        latent = np.random.default_rng(random_state).normal(
            size=(n_samples, self.dimension_)
        )
        return self.inverse_transform(latent)[0]


class AffineCouplingLayer:
    """A fixed Real-NVP-style affine coupling layer with exact inverse/log-Jacobian."""

    def __init__(
        self,
        dimension: int,
        mask: np.ndarray | None = None,
        scale_limit: float = 0.8,
        random_state: int = 550,
    ) -> None:
        if dimension < 2:
            raise ValueError("dimension must be at least two")
        self.dimension = int(dimension)
        default_mask = np.arange(dimension) % 2 == 0
        self.mask = default_mask if mask is None else np.asarray(mask, dtype=bool)
        if self.mask.shape != (dimension,) or self.mask.all() or (~self.mask).all():
            raise ValueError("mask must select a proper subset of dimensions")
        self.scale_limit = float(scale_limit)
        rng = np.random.default_rng(random_state)
        n_condition = int(self.mask.sum())
        n_transformed = dimension - n_condition
        self.scale_weights = rng.normal(scale=0.25, size=(n_condition, n_transformed))
        self.shift_weights = rng.normal(scale=0.25, size=(n_condition, n_transformed))
        self.scale_bias = rng.normal(scale=0.05, size=n_transformed)
        self.shift_bias = rng.normal(scale=0.05, size=n_transformed)

    def _scale_shift(self, conditioned: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = self.scale_limit * np.tanh(
            conditioned @ self.scale_weights + self.scale_bias
        )
        shift = conditioned @ self.shift_weights + self.shift_bias
        return scale, shift

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        output = X.copy()
        scale, shift = self._scale_shift(X[:, self.mask])
        output[:, ~self.mask] = X[:, ~self.mask] * np.exp(scale) + shift
        return output, scale.sum(axis=1)

    def inverse(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Y = np.asarray(Y, dtype=float)
        output = Y.copy()
        scale, shift = self._scale_shift(Y[:, self.mask])
        output[:, ~self.mask] = (Y[:, ~self.mask] - shift) * np.exp(-scale)
        return output, -scale.sum(axis=1)
