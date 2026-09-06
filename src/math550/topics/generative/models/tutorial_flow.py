"""PyTorch implementation of the two-moons flow-matching tutorial example."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .neural import resolve_torch_device


class TutorialVelocityMLP(nn.Module):
    """The three-hidden-layer, width-64 ELU network used in tutorial Code 1."""

    def __init__(self, dimension: int = 2, hidden_size: int = 64) -> None:
        super().__init__()
        if dimension < 1 or hidden_size < 1:
            raise ValueError("dimension and hidden_size must be positive")
        self.dimension = int(dimension)
        self.network = nn.Sequential(
            nn.Linear(self.dimension + 1, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, self.dimension),
        )

    def forward(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.dimension:
            raise ValueError("state has the wrong shape")
        if time.ndim == 1:
            time = time[:, None]
        if time.shape != (len(state), 1):
            raise ValueError("time must have shape (batch,) or (batch, 1)")
        return self.network(torch.cat([time, state], dim=1))


class TwoMoonsFlowMatcher:
    """Online flow matching and midpoint ODE sampling for a fixed 2D dataset."""

    def __init__(
        self,
        hidden_size: int = 64,
        learning_rate: float = 1e-2,
        updates: int = 10_000,
        random_state: int = 550,
        device: str = "auto",
    ) -> None:
        if learning_rate <= 0 or updates < 1:
            raise ValueError("learning_rate and updates must be positive")
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.updates = int(updates)
        self.random_state = int(random_state)
        self.device = str(device)
        resolve_torch_device(device)

    def fit(self, target: np.ndarray) -> "TwoMoonsFlowMatcher":
        target = np.asarray(target, dtype=np.float32)
        if target.ndim != 2 or target.shape[1] != 2 or len(target) == 0:
            raise ValueError("target must have shape (n, 2)")
        if not np.isfinite(target).all():
            raise ValueError("target contains NaN or infinity")

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.device_ = resolve_torch_device(self.device)
        self.network_ = TutorialVelocityMLP(2, self.hidden_size).to(self.device_)
        self.optimizer_ = torch.optim.Adam(
            self.network_.parameters(), lr=self.learning_rate
        )
        self.loss_function_ = nn.MSELoss()
        target_tensor = torch.as_tensor(target, device=self.device_)
        generator = torch.Generator(device="cpu").manual_seed(self.random_state + 1)
        self.loss_curve_: list[float] = []
        self.network_.train()
        for update in range(self.updates):
            noise = torch.randn(target_tensor.shape, generator=generator).to(self.device_)
            time = torch.rand((len(target_tensor), 1), generator=generator).to(
                self.device_
            )
            state = (1.0 - time) * noise + time * target_tensor
            target_velocity = target_tensor - noise
            self.optimizer_.zero_grad(set_to_none=True)
            loss = self.loss_function_(
                self.network_(time, state), target_velocity
            )
            loss.backward()
            self.optimizer_.step()
            if update == 0 or (update + 1) % 100 == 0:
                self.loss_curve_.append(float(loss.detach().cpu()))
        self.parameter_count_ = sum(p.numel() for p in self.network_.parameters())
        self.is_fitted_ = True
        return self

    def sample(
        self,
        n_samples: int = 300,
        n_steps: int = 8,
        random_state: int | None = None,
        return_path: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Integrate with the midpoint rule, matching tutorial Code 1."""

        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("fit must be called before sample")
        if n_samples < 1 or n_steps < 1:
            raise ValueError("n_samples and n_steps must be positive")
        seed = self.random_state + 2 if random_state is None else int(random_state)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        state = torch.randn((n_samples, 2), generator=generator).to(self.device_)
        step_size = 1.0 / n_steps
        path = [state.detach().cpu().numpy().copy()]
        self.network_.eval()
        with torch.no_grad():
            for step in range(n_steps):
                time = torch.full(
                    (n_samples, 1), step * step_size, device=self.device_
                )
                midpoint = state + 0.5 * step_size * self.network_(time, state)
                midpoint_time = time + 0.5 * step_size
                state = state + step_size * self.network_(midpoint_time, midpoint)
                path.append(state.detach().cpu().numpy().copy())
        samples = state.cpu().numpy().astype(float, copy=False)
        if return_path:
            return samples, np.stack(path)
        return samples
