"""Conditional flow matching with explicit target construction and ODE solvers."""

from __future__ import annotations

import numpy as np


def make_flow_matching_training_data(
    target_samples: np.ndarray,
    n_pairs: int,
    random_state: int = 550,
    sigma_min: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct independent-coupling conditional flow-matching regression pairs."""

    target_samples = np.asarray(target_samples, dtype=float)
    if target_samples.ndim != 2 or len(target_samples) == 0:
        raise ValueError("target_samples must be a non-empty matrix")
    if n_pairs < 1 or not 0.0 <= sigma_min < 1.0:
        raise ValueError("invalid n_pairs or sigma_min")
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(target_samples), size=n_pairs)
    data = target_samples[indices]
    noise = rng.normal(size=data.shape)
    time = rng.uniform(0.0, 1.0, size=(n_pairs, 1))
    path = (1.0 - (1.0 - sigma_min) * time) * noise + time * data
    velocity = data - (1.0 - sigma_min) * noise
    features = np.concatenate([time, path], axis=1)
    return features, velocity


class ConditionalFlowMatcher:
    """A vector-field regressor plus transparent Euler and RK4 samplers."""

    def __init__(
        self,
        regressor: object,
        sigma_min: float = 0.01,
        random_state: int = 550,
        name: str = "flow matching",
    ) -> None:
        if not 0.0 <= sigma_min < 1.0:
            raise ValueError("sigma_min must lie in [0, 1)")
        self.regressor = regressor
        self.sigma_min = float(sigma_min)
        self.random_state = int(random_state)
        self.name = str(name)

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        validation_features: np.ndarray | None = None,
        validation_targets: np.ndarray | None = None,
    ) -> "ConditionalFlowMatcher":
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets):
            raise ValueError("features and targets must be aligned matrices")
        if features.shape[1] != targets.shape[1] + 1:
            raise ValueError("flow features must contain time plus the state")
        self.dimension_ = targets.shape[1]
        self.regressor.fit(
            features,
            targets,
            validation_X=validation_features,
            validation_y=validation_targets,
        )
        self.is_fitted_ = True
        return self

    def velocity(self, state: np.ndarray, time: float) -> np.ndarray:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("fit must be called before velocity")
        state = np.asarray(state, dtype=float)
        if state.ndim != 2 or state.shape[1] != self.dimension_:
            raise ValueError("state has the wrong shape")
        times = np.full((len(state), 1), float(time))
        return np.asarray(
            self.regressor.predict(np.concatenate([times, state], axis=1)),
            dtype=float,
        )

    def sample(
        self,
        n_samples: int,
        n_steps: int = 40,
        random_state: int | None = None,
        solver: str = "rk4",
        return_path: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if n_samples < 1 or n_steps < 1:
            raise ValueError("n_samples and n_steps must be positive")
        if solver not in {"euler", "rk4"}:
            raise ValueError("solver must be 'euler' or 'rk4'")
        seed = self.random_state if random_state is None else int(random_state)
        state = np.random.default_rng(seed).normal(size=(n_samples, self.dimension_))
        snapshots = [state.copy()]
        step_size = 1.0 / n_steps
        for index in range(n_steps):
            time = index * step_size
            if solver == "euler":
                state = state + step_size * self.velocity(state, time)
            else:
                k1 = self.velocity(state, time)
                k2 = self.velocity(state + 0.5 * step_size * k1, time + 0.5 * step_size)
                k3 = self.velocity(state + 0.5 * step_size * k2, time + 0.5 * step_size)
                k4 = self.velocity(state + step_size * k3, time + step_size)
                state = state + step_size * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            if return_path:
                snapshots.append(state.copy())
        if not np.isfinite(state).all():
            raise FloatingPointError("ODE integration produced NaN or infinity")
        if return_path:
            return state, np.asarray(snapshots)
        return state

    def regression_mse(self, features: np.ndarray, targets: np.ndarray) -> float:
        prediction = np.asarray(self.regressor.predict(features), dtype=float)
        return float(np.mean((prediction - targets) ** 2))
