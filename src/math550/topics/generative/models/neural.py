"""PyTorch vector regressors for flow matching and diffusion."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


SUPPORTED_TORCH_DEVICES = ("auto", "cuda", "mps", "cpu")


def _mps_is_available() -> bool:
    """Return whether PyTorch exposes an available Apple Metal backend."""

    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_torch_device(requested: str = "auto") -> torch.device:
    """Resolve a requested device to CUDA, Apple Metal, or CPU.

    ``auto`` prefers CUDA, then Apple Metal (MPS), then CPU. Explicit requests
    for unavailable accelerators fail instead of silently changing hardware.
    """

    requested = str(requested).lower()
    if requested not in SUPPORTED_TORCH_DEVICES:
        choices = ", ".join(SUPPORTED_TORCH_DEVICES)
        raise ValueError(f"device must be one of: {choices}")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested == "mps" and not _mps_is_available():
        raise ValueError("MPS was requested but is not available")
    return torch.device(requested)


class TimeConditionedMLP(nn.Module):
    """Two-hidden-layer vector regressor built from standard PyTorch layers."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden_sizes: tuple[int, int] = (64, 64),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if n_inputs < 1 or n_outputs < 1:
            raise ValueError("input and output dimensions must be positive")
        if len(hidden_sizes) != 2 or min(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must contain two positive widths")
        activations: dict[str, type[nn.Module]] = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
        }
        if activation not in activations:
            raise ValueError("activation must be 'tanh' or 'relu'")
        activation_layer = activations[activation]
        self.network = nn.Sequential(
            nn.Linear(n_inputs, hidden_sizes[0]),
            activation_layer(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            activation_layer(),
            nn.Linear(hidden_sizes[1], n_outputs),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class TorchVectorRegressor:
    """Explicit PyTorch training and evaluation loop with a NumPy-facing API.

    ``model_kind='mlp'`` uses :class:`TimeConditionedMLP`. ``model_kind='linear'``
    is a capacity ablation trained by the same loss, optimizer, minibatches, and
    epoch budget. PyTorch autograd supplies all parameter derivatives.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, int] = (64, 64),
        activation: str = "tanh",
        model_kind: str = "mlp",
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 120,
        weight_decay: float = 0.0,
        random_state: int = 550,
        device: str = "auto",
    ) -> None:
        if model_kind not in {"mlp", "linear"}:
            raise ValueError("model_kind must be 'mlp' or 'linear'")
        if len(hidden_sizes) != 2 or min(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must contain two positive widths")
        if learning_rate <= 0 or batch_size < 1 or max_epochs < 1:
            raise ValueError("invalid optimization hyperparameter")
        if weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        resolve_torch_device(device)
        self.hidden_sizes = tuple(int(value) for value in hidden_sizes)
        self.activation = str(activation)
        self.model_kind = str(model_kind)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.device = str(device)

    @staticmethod
    def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or len(array) == 0:
            raise ValueError(f"{name} must be a non-empty two-dimensional array")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinity")
        return array

    def _build_network(self, n_inputs: int, n_outputs: int) -> nn.Module:
        if self.model_kind == "linear":
            return nn.Linear(n_inputs, n_outputs)
        return TimeConditionedMLP(
            n_inputs,
            n_outputs,
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_X: np.ndarray | None = None,
        validation_y: np.ndarray | None = None,
    ) -> "TorchVectorRegressor":
        X = self._as_2d(X, "X")
        y = self._as_2d(y, "y")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        if (validation_X is None) != (validation_y is None):
            raise ValueError("validation_X and validation_y must be supplied together")

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = y.shape[1]
        self.device_ = resolve_torch_device(self.device)
        self.network_ = self._build_network(self.n_features_in_, self.n_outputs_).to(
            self.device_
        )
        self.loss_function_ = nn.MSELoss(reduction="mean")
        self.optimizer_ = torch.optim.Adam(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        inputs = torch.as_tensor(X, dtype=torch.float32, device=self.device_)
        targets = torch.as_tensor(y, dtype=torch.float32, device=self.device_)
        if validation_X is not None and validation_y is not None:
            validation_X = self._as_2d(validation_X, "validation_X")
            validation_y = self._as_2d(validation_y, "validation_y")
            if validation_X.shape[1] != X.shape[1] or validation_y.shape[1] != y.shape[1]:
                raise ValueError("validation arrays have incompatible columns")
            validation_inputs = torch.as_tensor(
                validation_X, dtype=torch.float32, device=self.device_
            )
            validation_targets = torch.as_tensor(
                validation_y, dtype=torch.float32, device=self.device_
            )
        else:
            validation_inputs = validation_targets = None

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_state + 1)
        self.loss_curve_: list[float] = []
        self.validation_loss_curve_: list[float] = []
        for _ in range(self.max_epochs):
            self.network_.train()
            ordering = torch.randperm(len(inputs), generator=generator)
            epoch_loss = 0.0
            for start in range(0, len(inputs), self.batch_size):
                batch = ordering[start : start + self.batch_size].to(self.device_)
                self.optimizer_.zero_grad(set_to_none=True)
                prediction = self.network_(inputs[batch])
                loss = self.loss_function_(prediction, targets[batch])
                loss.backward()
                self.optimizer_.step()
                epoch_loss += float(loss.detach().cpu()) * len(batch)
            self.loss_curve_.append(epoch_loss / len(inputs))
            if validation_inputs is not None and validation_targets is not None:
                self.network_.eval()
                with torch.no_grad():
                    validation_loss = self.loss_function_(
                        self.network_(validation_inputs), validation_targets
                    )
                self.validation_loss_curve_.append(float(validation_loss.cpu()))

        self.parameter_count_ = sum(
            parameter.numel() for parameter in self.network_.parameters()
        )
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("fit must be called before predict")
        X = self._as_2d(X, "X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of columns")
        self.network_.eval()
        with torch.no_grad():
            prediction = self.network_(
                torch.as_tensor(X, dtype=torch.float32, device=self.device_)
            )
        return prediction.cpu().numpy().astype(float, copy=False)
