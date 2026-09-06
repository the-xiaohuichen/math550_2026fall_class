"""Denoising diffusion target construction and a deterministic DDIM sampler."""

from __future__ import annotations

import numpy as np


def linear_beta_schedule(
    n_steps: int = 50, beta_start: float = 1e-4, beta_end: float = 0.02
) -> np.ndarray:
    if n_steps < 2 or not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError("invalid diffusion schedule")
    return np.linspace(beta_start, beta_end, n_steps, dtype=float)


def make_diffusion_training_data(
    data_samples: np.ndarray,
    n_pairs: int,
    betas: np.ndarray,
    random_state: int = 550,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct epsilon-prediction targets for a variance-preserving diffusion."""

    data_samples = np.asarray(data_samples, dtype=float)
    betas = np.asarray(betas, dtype=float)
    if data_samples.ndim != 2 or len(data_samples) == 0 or n_pairs < 1:
        raise ValueError("invalid data_samples or n_pairs")
    if betas.ndim != 1 or len(betas) < 2 or np.any((betas <= 0) | (betas >= 1)):
        raise ValueError("betas must be a vector in (0, 1)")
    alpha_bars = np.cumprod(1.0 - betas)
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(data_samples), size=n_pairs)
    steps = rng.integers(0, len(betas), size=n_pairs)
    clean = data_samples[indices]
    noise = rng.normal(size=clean.shape)
    retained = np.sqrt(alpha_bars[steps])[:, None]
    removed = np.sqrt(1.0 - alpha_bars[steps])[:, None]
    noised = retained * clean + removed * noise
    time = ((steps + 1) / len(betas))[:, None]
    return np.concatenate([time, noised], axis=1), noise


class DenoisingDiffusion:
    """An epsilon-prediction network with a transparent deterministic sampler."""

    def __init__(
        self,
        regressor: object,
        betas: np.ndarray,
        random_state: int = 550,
        name: str = "diffusion",
    ) -> None:
        self.betas = np.asarray(betas, dtype=float)
        if self.betas.ndim != 1 or len(self.betas) < 2:
            raise ValueError("betas must be a one-dimensional schedule")
        self.alpha_bars_ = np.cumprod(1.0 - self.betas)
        self.regressor = regressor
        self.random_state = int(random_state)
        self.name = str(name)

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        validation_features: np.ndarray | None = None,
        validation_targets: np.ndarray | None = None,
    ) -> "DenoisingDiffusion":
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets):
            raise ValueError("features and targets must be aligned matrices")
        if features.shape[1] != targets.shape[1] + 1:
            raise ValueError("diffusion features must contain time plus the state")
        self.dimension_ = targets.shape[1]
        self.regressor.fit(
            features,
            targets,
            validation_X=validation_features,
            validation_y=validation_targets,
        )
        self.is_fitted_ = True
        return self

    def predict_noise(self, state: np.ndarray, step: int) -> np.ndarray:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("fit must be called before predict_noise")
        time = np.full((len(state), 1), (step + 1) / len(self.betas))
        features = np.concatenate([time, state], axis=1)
        return np.asarray(self.regressor.predict(features), dtype=float)

    def _sampling_indices(self, n_sampling_steps: int) -> np.ndarray:
        if not 2 <= n_sampling_steps <= len(self.betas):
            raise ValueError("n_sampling_steps must be between 2 and schedule length")
        indices = np.linspace(0, len(self.betas) - 1, n_sampling_steps)
        return np.unique(np.rint(indices).astype(int))[::-1]

    def sample(
        self,
        n_samples: int,
        n_sampling_steps: int | None = None,
        random_state: int | None = None,
        return_path: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        n_sampling_steps = n_sampling_steps or len(self.betas)
        indices = self._sampling_indices(n_sampling_steps)
        seed = self.random_state if random_state is None else int(random_state)
        state = np.random.default_rng(seed).normal(size=(n_samples, self.dimension_))
        snapshots = [state.copy()]
        for position, step in enumerate(indices):
            alpha_bar = self.alpha_bars_[step]
            predicted_noise = self.predict_noise(state, int(step))
            predicted_clean = (
                state - np.sqrt(1.0 - alpha_bar) * predicted_noise
            ) / np.sqrt(alpha_bar)
            predicted_clean = np.clip(predicted_clean, -5.0, 5.0)
            if position == len(indices) - 1:
                state = predicted_clean
            else:
                previous_step = indices[position + 1]
                previous_alpha_bar = self.alpha_bars_[previous_step]
                state = (
                    np.sqrt(previous_alpha_bar) * predicted_clean
                    + np.sqrt(1.0 - previous_alpha_bar) * predicted_noise
                )
            if return_path:
                snapshots.append(state.copy())
        if not np.isfinite(state).all():
            raise FloatingPointError("diffusion sampling produced NaN or infinity")
        if return_path:
            return state, np.asarray(snapshots)
        return state

    def regression_mse(self, features: np.ndarray, targets: np.ndarray) -> float:
        prediction = np.asarray(self.regressor.predict(features), dtype=float)
        return float(np.mean((prediction - targets) ** 2))
