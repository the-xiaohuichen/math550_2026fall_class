"""PyTorch variational autoencoders for vector data and 28-by-28 MNIST."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .neural import resolve_torch_device


def diagonal_gaussian_kl(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """Return per-example KL(q||N(0,I)) for a diagonal Gaussian posterior."""

    if mean.shape != log_variance.shape or mean.ndim != 2:
        raise ValueError("mean and log_variance must be aligned matrices")
    return 0.5 * torch.sum(
        mean.square() + log_variance.exp() - 1.0 - log_variance, dim=1
    )


class VectorVAE(nn.Module):
    """Small Gaussian VAE composed only from standard PyTorch layers."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 2,
        hidden_sizes: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        if input_dim < 1 or not 1 <= latent_dim <= input_dim:
            raise ValueError("latent_dim must lie between one and input_dim")
        if len(hidden_sizes) != 2 or min(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must contain two positive widths")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        first, second = (int(value) for value in hidden_sizes)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, first),
            nn.ELU(),
            nn.Linear(first, second),
            nn.ELU(),
        )
        self.posterior_mean = nn.Linear(second, self.latent_dim)
        self.posterior_log_variance = nn.Linear(second, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, second),
            nn.ELU(),
            nn.Linear(second, first),
            nn.ELU(),
            nn.Linear(first, self.input_dim),
        )

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(values)
        mean = self.posterior_mean(encoded)
        log_variance = self.posterior_log_variance(encoded).clamp(-12.0, 8.0)
        return mean, log_variance

    @staticmethod
    def reparameterize(
        mean: torch.Tensor, log_variance: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        if mean.shape != log_variance.shape or mean.shape != noise.shape:
            raise ValueError("posterior parameters and noise must have matching shapes")
        return mean + torch.exp(0.5 * log_variance) * noise

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(
        self, values: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(values)
        latent = self.reparameterize(mean, log_variance, noise)
        return self.decode(latent), mean, log_variance


class GaussianVAE:
    """Explicit ELBO training loop for continuous vector observations.

    The decoder likelihood is an isotropic Gaussian with fixed ``observation_sigma``.
    The reported reconstruction cost omits the parameter-independent Gaussian
    normalizing constant, so the objective is the negative ELBO up to that constant.
    """

    def __init__(
        self,
        latent_dim: int = 2,
        hidden_sizes: tuple[int, int] = (64, 64),
        observation_sigma: float = 0.1,
        beta: float = 1.0,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 300,
        random_state: int = 550,
        device: str = "auto",
    ) -> None:
        if latent_dim < 1 or min(hidden_sizes) < 1:
            raise ValueError("latent and hidden dimensions must be positive")
        if observation_sigma <= 0 or beta < 0:
            raise ValueError("observation_sigma must be positive and beta nonnegative")
        if learning_rate <= 0 or batch_size < 1 or max_epochs < 1:
            raise ValueError("invalid optimization hyperparameter")
        resolve_torch_device(device)
        self.latent_dim = int(latent_dim)
        self.hidden_sizes = tuple(int(value) for value in hidden_sizes)
        self.observation_sigma = float(observation_sigma)
        self.beta = float(beta)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.random_state = int(random_state)
        self.device = str(device)

    @staticmethod
    def _validate(values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or len(array) == 0:
            raise ValueError(f"{name} must be a non-empty matrix")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinity")
        return array

    def _components(
        self,
        values: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction, mean, log_variance = self.network_(values, noise)
        reconstruction_cost = 0.5 * torch.sum(
            ((values - reconstruction) / self.observation_sigma).square(), dim=1
        )
        kl = diagonal_gaussian_kl(mean, log_variance)
        objective = reconstruction_cost + self.beta * kl
        return objective, reconstruction_cost, kl

    def _evaluate_components(
        self, values: np.ndarray, random_state: int
    ) -> dict[str, float]:
        values = self._validate(values, "values")
        if values.shape[1] != self.input_dim_:
            raise ValueError("values have the wrong number of columns")
        values_cpu = torch.from_numpy(values)
        generator = torch.Generator(device="cpu").manual_seed(int(random_state))
        totals = np.zeros(3, dtype=float)
        self.network_.eval()
        with torch.no_grad():
            for start in range(0, len(values_cpu), self.batch_size):
                batch = values_cpu[start : start + self.batch_size].to(self.device_)
                noise = torch.randn(
                    (len(batch), self.latent_dim), generator=generator
                ).to(self.device_)
                objective, reconstruction, kl = self._components(batch, noise)
                totals += np.array(
                    [objective.sum().cpu(), reconstruction.sum().cpu(), kl.sum().cpu()],
                    dtype=float,
                )
        return {
            "objective": float(totals[0] / len(values_cpu)),
            "reconstruction_cost": float(totals[1] / len(values_cpu)),
            "kl": float(totals[2] / len(values_cpu)),
        }

    def fit(
        self,
        values: np.ndarray,
        validation_values: np.ndarray | None = None,
    ) -> "GaussianVAE":
        values = self._validate(values, "values")
        if self.latent_dim > values.shape[1]:
            raise ValueError("latent_dim cannot exceed the observation dimension")
        if validation_values is not None:
            validation_values = self._validate(validation_values, "validation_values")
            if validation_values.shape[1] != values.shape[1]:
                raise ValueError("validation_values have incompatible columns")

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.device_ = resolve_torch_device(self.device)
        self.input_dim_ = int(values.shape[1])
        self.network_ = VectorVAE(
            self.input_dim_, self.latent_dim, self.hidden_sizes
        ).to(self.device_)
        self.optimizer_ = torch.optim.Adam(
            self.network_.parameters(), lr=self.learning_rate
        )
        values_cpu = torch.from_numpy(values)
        generator = torch.Generator(device="cpu").manual_seed(self.random_state + 1)
        self.loss_curve_: list[float] = []
        self.reconstruction_curve_: list[float] = []
        self.kl_curve_: list[float] = []
        self.validation_loss_curve_: list[float] = []

        for epoch in range(self.max_epochs):
            self.network_.train()
            order = torch.randperm(len(values_cpu), generator=generator)
            totals = np.zeros(3, dtype=float)
            for start in range(0, len(order), self.batch_size):
                indices = order[start : start + self.batch_size]
                batch = values_cpu[indices].to(self.device_)
                noise = torch.randn(
                    (len(batch), self.latent_dim), generator=generator
                ).to(self.device_)
                self.optimizer_.zero_grad(set_to_none=True)
                objective, reconstruction, kl = self._components(batch, noise)
                loss = objective.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network_.parameters(), 10.0)
                self.optimizer_.step()
                totals += np.array(
                    [objective.sum().detach().cpu(), reconstruction.sum().detach().cpu(), kl.sum().detach().cpu()],
                    dtype=float,
                )
            self.loss_curve_.append(float(totals[0] / len(values_cpu)))
            self.reconstruction_curve_.append(float(totals[1] / len(values_cpu)))
            self.kl_curve_.append(float(totals[2] / len(values_cpu)))
            if validation_values is not None:
                validation = self._evaluate_components(
                    validation_values, self.random_state + 10_000
                )
                self.validation_loss_curve_.append(validation["objective"])

        self.parameter_count_ = sum(p.numel() for p in self.network_.parameters())
        self.n_train_rows_ = len(values_cpu)
        self.is_fitted_ = True
        return self

    def evaluate(self, values: np.ndarray) -> dict[str, float]:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("fit must be called before evaluate")
        result = self._evaluate_components(values, self.random_state + 20_000)
        result["reconstruction_mse"] = self.reconstruction_mse(values)
        return result

    def reconstruction_mse(self, values: np.ndarray) -> float:
        values = self._validate(values, "values")
        self.network_.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(values).to(self.device_)
            mean, _ = self.network_.encode(tensor)
            reconstruction = self.network_.decode(mean)
            return float(F.mse_loss(reconstruction, tensor).cpu())

    def encode(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior mean and standard deviation for inspection."""

        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("fit must be called before encode")
        values = self._validate(values, "values")
        self.network_.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(values).to(self.device_)
            mean, log_variance = self.network_.encode(tensor)
        return (
            mean.cpu().numpy().astype(float, copy=False),
            torch.exp(0.5 * log_variance).cpu().numpy().astype(float, copy=False),
        )

    def reconstruct(self, values: np.ndarray) -> np.ndarray:
        """Decode the posterior mean for a deterministic reconstruction."""

        mean, _ = self.encode(values)
        self.network_.eval()
        with torch.no_grad():
            reconstruction = self.network_.decode(
                torch.as_tensor(mean, dtype=torch.float32, device=self.device_)
            )
        return reconstruction.cpu().numpy().astype(float, copy=False)

    def sample(
        self,
        n_samples: int,
        random_state: int | None = None,
        sample_observation: bool = True,
        return_latent: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("fit must be called before sample")
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        seed = self.random_state + 2 if random_state is None else int(random_state)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latent_cpu = torch.randn((n_samples, self.latent_dim), generator=generator)
        self.network_.eval()
        with torch.no_grad():
            decoded = self.network_.decode(latent_cpu.to(self.device_)).cpu()
            if sample_observation:
                decoded = decoded + self.observation_sigma * torch.randn(
                    decoded.shape, generator=generator
                )
        samples = decoded.numpy().astype(float, copy=False)
        latent = latent_cpu.numpy().astype(float, copy=False)
        if return_latent:
            return samples, latent
        return samples


class ConditionalMNISTVAEModel(nn.Module):
    """Compact class-conditional convolutional VAE for 28-by-28 images."""

    def __init__(
        self, latent_dim: int = 16, hidden_size: int = 16, n_classes: int = 10
    ) -> None:
        super().__init__()
        if latent_dim < 1 or hidden_size < 4 or n_classes < 2:
            raise ValueError("invalid latent, hidden, or class dimension")
        self.latent_dim = int(latent_dim)
        self.hidden_size = int(hidden_size)
        self.n_classes = int(n_classes)
        width = self.hidden_size
        groups = 4 if width % 4 == 0 else 1
        self.encoder = nn.Sequential(
            nn.Conv2d(1 + self.n_classes, width, 4, stride=2, padding=1),
            nn.GroupNorm(groups, width),
            nn.SiLU(),
            nn.Conv2d(width, 2 * width, 4, stride=2, padding=1),
            nn.GroupNorm(groups, 2 * width),
            nn.SiLU(),
        )
        encoded_dim = 2 * width * 7 * 7
        self.encoder_hidden = nn.Sequential(nn.Linear(encoded_dim, 128), nn.SiLU())
        self.posterior_mean = nn.Linear(128, self.latent_dim)
        self.posterior_log_variance = nn.Linear(128, self.latent_dim)
        self.decoder_hidden = nn.Sequential(
            nn.Linear(self.latent_dim + self.n_classes, 128),
            nn.SiLU(),
            nn.Linear(128, encoded_dim),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2 * width, width, 4, stride=2, padding=1),
            nn.GroupNorm(groups, width),
            nn.SiLU(),
            nn.ConvTranspose2d(width, 1, 4, stride=2, padding=1),
        )

    def _one_hot(self, labels: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return F.one_hot(labels.long(), num_classes=self.n_classes).to(dtype)

    def encode(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        class_code = self._one_hot(labels, images.dtype)
        class_maps = class_code[:, :, None, None].expand(-1, -1, 28, 28)
        hidden = self.encoder(torch.cat([images, class_maps], dim=1)).flatten(1)
        hidden = self.encoder_hidden(hidden)
        return (
            self.posterior_mean(hidden),
            self.posterior_log_variance(hidden).clamp(-12.0, 8.0),
        )

    def decode(self, latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        class_code = self._one_hot(labels, latent.dtype)
        width = self.hidden_size
        hidden = self.decoder_hidden(torch.cat([latent, class_code], dim=1))
        return self.decoder(hidden.reshape(len(latent), 2 * width, 7, 7))

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(images, labels)
        latent = VectorVAE.reparameterize(mean, log_variance, noise)
        return self.decode(latent, labels), mean, log_variance


class ConditionalMNISTVAE:
    """Class-conditional Bernoulli VAE with explicit PyTorch training loops."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_size: int = 16,
        beta: float = 1.0,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 8,
        random_state: int = 550,
        device: str = "auto",
    ) -> None:
        if latent_dim < 1 or hidden_size < 4 or beta < 0:
            raise ValueError("invalid architecture or beta")
        if learning_rate <= 0 or batch_size < 1 or max_epochs < 1:
            raise ValueError("invalid optimization hyperparameter")
        resolve_torch_device(device)
        self.latent_dim = int(latent_dim)
        self.hidden_size = int(hidden_size)
        self.beta = float(beta)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.random_state = int(random_state)
        self.device = str(device)

    @staticmethod
    def _validate(images: np.ndarray, labels: np.ndarray, name: str) -> None:
        if images.ndim != 3 or images.shape[1:] != (28, 28) or len(images) == 0:
            raise ValueError(f"{name} images must have shape (n, 28, 28)")
        if images.dtype != np.uint8:
            raise ValueError(f"{name} images must use uint8 pixels")
        if labels.shape != (len(images),) or labels.min() < 0 or labels.max() > 9:
            raise ValueError(f"{name} labels must be aligned values in 0,...,9")

    def _components(
        self, images: torch.Tensor, labels: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, mean, log_variance = self.network_(images, labels, noise)
        reconstruction = F.binary_cross_entropy_with_logits(
            logits, images, reduction="none"
        ).flatten(1).sum(dim=1)
        kl = diagonal_gaussian_kl(mean, log_variance)
        return reconstruction + self.beta * kl, reconstruction, kl

    def _evaluate_components(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        max_rows: int | None,
        random_state: int,
    ) -> dict[str, float]:
        self._validate(images, labels, "evaluation")
        n_rows = len(images) if max_rows is None else min(max_rows, len(images))
        generator = torch.Generator(device="cpu").manual_seed(int(random_state))
        totals = np.zeros(3, dtype=float)
        self.network_.eval()
        with torch.no_grad():
            for start in range(0, n_rows, self.batch_size):
                stop = min(start + self.batch_size, n_rows)
                batch = torch.from_numpy(images[start:stop]).to(torch.float32).div(255.0)
                batch = batch.unsqueeze(1).to(self.device_)
                label_batch = torch.from_numpy(
                    labels[start:stop].astype(np.int64, copy=False)
                ).to(self.device_)
                noise = torch.randn(
                    (len(batch), self.latent_dim), generator=generator
                ).to(self.device_)
                objective, reconstruction, kl = self._components(
                    batch, label_batch, noise
                )
                totals += np.array(
                    [objective.sum().cpu(), reconstruction.sum().cpu(), kl.sum().cpu()],
                    dtype=float,
                )
        return {
            "negative_elbo": float(totals[0] / n_rows),
            "reconstruction_bce": float(totals[1] / n_rows),
            "kl": float(totals[2] / n_rows),
        }

    def fit(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        validation_images: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
        max_train_rows: int | None = None,
    ) -> "ConditionalMNISTVAE":
        train_images = np.asarray(train_images)
        train_labels = np.asarray(train_labels)
        self._validate(train_images, train_labels, "training")
        if (validation_images is None) != (validation_labels is None):
            raise ValueError("validation images and labels must be supplied together")
        if validation_images is not None and validation_labels is not None:
            validation_images = np.asarray(validation_images)
            validation_labels = np.asarray(validation_labels)
            self._validate(validation_images, validation_labels, "validation")
        if max_train_rows is not None:
            if max_train_rows < 1:
                raise ValueError("max_train_rows must be positive")
            train_images = train_images[:max_train_rows]
            train_labels = train_labels[:max_train_rows]

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.device_ = resolve_torch_device(self.device)
        self.network_ = ConditionalMNISTVAEModel(
            self.latent_dim, self.hidden_size
        ).to(self.device_)
        self.optimizer_ = torch.optim.AdamW(
            self.network_.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        images_cpu = torch.from_numpy(train_images)
        labels_cpu = torch.from_numpy(train_labels.astype(np.int64, copy=False))
        generator = torch.Generator(device="cpu").manual_seed(self.random_state + 1)
        self.loss_curve_: list[float] = []
        self.reconstruction_curve_: list[float] = []
        self.kl_curve_: list[float] = []
        self.validation_loss_curve_: list[float] = []

        for _ in range(self.max_epochs):
            self.network_.train()
            order = torch.randperm(len(images_cpu), generator=generator)
            totals = np.zeros(3, dtype=float)
            for start in range(0, len(order), self.batch_size):
                indices = order[start : start + self.batch_size]
                images = images_cpu[indices].to(torch.float32).div(255.0)
                images = images.unsqueeze(1).to(self.device_)
                labels = labels_cpu[indices].to(self.device_)
                noise = torch.randn(
                    (len(images), self.latent_dim), generator=generator
                ).to(self.device_)
                self.optimizer_.zero_grad(set_to_none=True)
                objective, reconstruction, kl = self._components(images, labels, noise)
                objective.mean().backward()
                nn.utils.clip_grad_norm_(self.network_.parameters(), 10.0)
                self.optimizer_.step()
                totals += np.array(
                    [objective.sum().detach().cpu(), reconstruction.sum().detach().cpu(), kl.sum().detach().cpu()],
                    dtype=float,
                )
            self.loss_curve_.append(float(totals[0] / len(images_cpu)))
            self.reconstruction_curve_.append(float(totals[1] / len(images_cpu)))
            self.kl_curve_.append(float(totals[2] / len(images_cpu)))
            if validation_images is not None and validation_labels is not None:
                evaluation = self._evaluate_components(
                    validation_images,
                    validation_labels,
                    max_rows=min(2_000, len(validation_images)),
                    random_state=self.random_state + 10_000,
                )
                self.validation_loss_curve_.append(evaluation["negative_elbo"])

        self.parameter_count_ = sum(p.numel() for p in self.network_.parameters())
        self.n_train_rows_ = len(images_cpu)
        self.is_fitted_ = True
        return self

    def evaluate(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        max_rows: int | None = None,
    ) -> dict[str, float]:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("fit must be called before evaluate")
        return self._evaluate_components(
            np.asarray(images),
            np.asarray(labels),
            max_rows,
            self.random_state + 20_000,
        )

    def reconstruct(self, images: np.ndarray, labels: np.ndarray) -> np.ndarray:
        images = np.asarray(images)
        labels = np.asarray(labels)
        self._validate(images, labels, "reconstruction")
        self.network_.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(images).to(torch.float32).div(255.0)
            tensor = tensor.unsqueeze(1).to(self.device_)
            label_tensor = torch.from_numpy(labels.astype(np.int64, copy=False)).to(
                self.device_
            )
            mean, _ = self.network_.encode(tensor, label_tensor)
            logits = self.network_.decode(mean, label_tensor)
        return torch.sigmoid(logits).cpu().numpy()[:, 0].astype(float, copy=False)

    def sample(
        self, labels: np.ndarray, random_state: int | None = None
    ) -> np.ndarray:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("fit must be called before sample")
        labels = np.asarray(labels, dtype=np.int64)
        if labels.ndim != 1 or len(labels) == 0 or labels.min() < 0 or labels.max() > 9:
            raise ValueError("labels must be a non-empty vector in 0,...,9")
        seed = self.random_state + 2 if random_state is None else int(random_state)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latent = torch.randn((len(labels), self.latent_dim), generator=generator).to(
            self.device_
        )
        label_tensor = torch.from_numpy(labels).to(self.device_)
        self.network_.eval()
        with torch.no_grad():
            logits = self.network_.decode(latent, label_tensor)
        return torch.sigmoid(logits).cpu().numpy()[:, 0].astype(float, copy=False)

    def state_dict(self) -> dict[str, object]:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("fit must be called before state_dict")
        return {
            "network": self.network_.state_dict(),
            "config": {
                "latent_dim": self.latent_dim,
                "hidden_size": self.hidden_size,
                "beta": self.beta,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "random_state": self.random_state,
                "device": self.device,
            },
            "loss_curve": self.loss_curve_,
            "reconstruction_curve": self.reconstruction_curve_,
            "kl_curve": self.kl_curve_,
            "validation_loss_curve": self.validation_loss_curve_,
            "parameter_count": self.parameter_count_,
            "n_train_rows": self.n_train_rows_,
        }

    @classmethod
    def from_state_dict(
        cls, payload: dict[str, object], device: str = "auto"
    ) -> "ConditionalMNISTVAE":
        config = dict(payload["config"])
        config["device"] = device
        model = cls(**config)
        model.device_ = resolve_torch_device(device)
        model.network_ = ConditionalMNISTVAEModel(
            model.latent_dim, model.hidden_size
        ).to(model.device_)
        model.network_.load_state_dict(payload["network"])
        model.network_.eval()
        model.loss_curve_ = [float(value) for value in payload["loss_curve"]]
        model.reconstruction_curve_ = [
            float(value) for value in payload["reconstruction_curve"]
        ]
        model.kl_curve_ = [float(value) for value in payload["kl_curve"]]
        model.validation_loss_curve_ = [
            float(value) for value in payload["validation_loss_curve"]
        ]
        model.parameter_count_ = int(payload["parameter_count"])
        model.n_train_rows_ = int(payload["n_train_rows"])
        model.is_fitted_ = True
        return model
