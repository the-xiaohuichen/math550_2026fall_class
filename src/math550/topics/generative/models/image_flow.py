"""Class-conditional PyTorch flow matching for original 28-by-28 MNIST."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .neural import resolve_torch_device


class MNISTVelocityCNN(nn.Module):
    """A compact U-Net-style velocity network using standard PyTorch layers."""

    def __init__(self, hidden_size: int = 16, n_classes: int = 10) -> None:
        super().__init__()
        if hidden_size < 1 or n_classes < 2:
            raise ValueError("hidden_size and n_classes must be positive")
        self.hidden_size = int(hidden_size)
        self.n_classes = int(n_classes)
        condition_channels = 1 + 1 + self.n_classes
        width = self.hidden_size
        self.encoder_14 = nn.Sequential(
            nn.Conv2d(condition_channels, width, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4 if width >= 4 else 1, width),
            nn.SiLU(),
        )
        self.encoder_7 = nn.Sequential(
            nn.Conv2d(width, 2 * width, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4 if width >= 4 else 1, 2 * width),
            nn.SiLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1),
            nn.GroupNorm(4 if width >= 4 else 1, 4 * width),
            nn.SiLU(),
        )
        self.up_14 = nn.ConvTranspose2d(
            4 * width, 2 * width, kernel_size=4, stride=2, padding=1
        )
        self.decoder_14 = nn.Sequential(
            nn.Conv2d(3 * width, 2 * width, kernel_size=3, padding=1),
            nn.GroupNorm(4 if width >= 4 else 1, 2 * width),
            nn.SiLU(),
        )
        self.up_28 = nn.ConvTranspose2d(
            2 * width, width, kernel_size=4, stride=2, padding=1
        )
        self.decoder_28 = nn.Sequential(
            nn.Conv2d(width + condition_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(4 if width >= 4 else 1, width),
            nn.SiLU(),
            nn.Conv2d(width, 1, kernel_size=3, padding=1),
        )

    def forward(
        self, time: torch.Tensor, images: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError("images must have shape (batch, 1, 28, 28)")
        if time.ndim == 1:
            time = time[:, None]
        if time.shape != (len(images), 1) or labels.shape != (len(images),):
            raise ValueError("time or labels have an incompatible shape")
        class_code = F.one_hot(labels.long(), num_classes=self.n_classes).to(images.dtype)
        class_maps = class_code[:, :, None, None].expand(-1, -1, 28, 28)
        time_map = time[:, :, None, None].expand(-1, -1, 28, 28)
        conditioned = torch.cat([images, time_map, class_maps], dim=1)
        encoded_14 = self.encoder_14(conditioned)
        encoded_7 = self.encoder_7(encoded_14)
        bottleneck = self.bottleneck(encoded_7)
        decoded_14 = self.decoder_14(
            torch.cat([self.up_14(bottleneck), encoded_14], dim=1)
        )
        decoded_28 = self.up_28(decoded_14)
        return self.decoder_28(torch.cat([decoded_28, conditioned], dim=1))


class ConditionalMNISTFlowMatcher:
    """Explicit conditional flow-matching training and midpoint sampling loops."""

    def __init__(
        self,
        hidden_size: int = 16,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 8,
        ema_decay: float = 0.995,
        random_state: int = 550,
        device: str = "auto",
    ) -> None:
        if learning_rate <= 0 or batch_size < 1 or max_epochs < 1:
            raise ValueError("invalid optimization hyperparameter")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.ema_decay = float(ema_decay)
        self.random_state = int(random_state)
        self.device = str(device)
        resolve_torch_device(device)

    @staticmethod
    def _validate(images: np.ndarray, labels: np.ndarray, name: str) -> None:
        if images.ndim != 3 or images.shape[1:] != (28, 28) or len(images) == 0:
            raise ValueError(f"{name} images must have shape (n, 28, 28)")
        if images.dtype != np.uint8:
            raise ValueError(f"{name} images must use uint8 pixels")
        if labels.shape != (len(images),):
            raise ValueError(f"{name} labels have the wrong shape")
        if np.min(labels) < 0 or np.max(labels) > 9:
            raise ValueError(f"{name} labels must lie between 0 and 9")

    @staticmethod
    def _scale(images: torch.Tensor) -> torch.Tensor:
        return images.to(torch.float32).div(127.5).sub(1.0)

    @torch.no_grad()
    def _update_ema(self) -> None:
        for averaged, current in zip(
            self.ema_network_.parameters(), self.network_.parameters()
        ):
            averaged.mul_(self.ema_decay).add_(current, alpha=1.0 - self.ema_decay)

    def fit(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        validation_images: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
        max_train_rows: int | None = None,
    ) -> "ConditionalMNISTFlowMatcher":
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
        self.network_ = MNISTVelocityCNN(self.hidden_size).to(self.device_)
        self.ema_network_ = copy.deepcopy(self.network_).eval()
        for parameter in self.ema_network_.parameters():
            parameter.requires_grad_(False)
        self.optimizer_ = torch.optim.AdamW(
            self.network_.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        self.loss_function_ = nn.MSELoss()
        images_cpu = torch.from_numpy(train_images)
        labels_cpu = torch.from_numpy(train_labels.astype(np.int64, copy=False))
        generator = torch.Generator(device="cpu").manual_seed(self.random_state + 1)
        self.loss_curve_: list[float] = []
        self.validation_loss_curve_: list[float] = []

        for _ in range(self.max_epochs):
            self.network_.train()
            order = torch.randperm(len(images_cpu), generator=generator)
            total_loss = 0.0
            for start in range(0, len(order), self.batch_size):
                indices = order[start : start + self.batch_size]
                target = self._scale(images_cpu[indices]).unsqueeze(1).to(self.device_)
                labels = labels_cpu[indices].to(self.device_)
                noise = torch.randn(target.shape, generator=generator).to(self.device_)
                time = torch.rand((len(target), 1), generator=generator).to(self.device_)
                time_image = time[:, :, None, None]
                state = (1.0 - time_image) * noise + time_image * target
                target_velocity = target - noise
                self.optimizer_.zero_grad(set_to_none=True)
                prediction = self.network_(time, state, labels)
                loss = self.loss_function_(prediction, target_velocity)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network_.parameters(), 5.0)
                self.optimizer_.step()
                self._update_ema()
                total_loss += float(loss.detach().cpu()) * len(indices)
            self.loss_curve_.append(total_loss / len(images_cpu))
            if validation_images is not None and validation_labels is not None:
                self.validation_loss_curve_.append(
                    self.objective_mse(
                        validation_images,
                        validation_labels,
                        max_rows=min(2_000, len(validation_images)),
                        random_state=self.random_state + 30,
                    )
                )

        self.parameter_count_ = sum(p.numel() for p in self.network_.parameters())
        self.n_train_rows_ = len(images_cpu)
        self.is_fitted_ = True
        return self

    def objective_mse(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        max_rows: int | None = None,
        random_state: int | None = None,
    ) -> float:
        network = getattr(self, "ema_network_", None)
        if network is None:
            raise RuntimeError("fit must be called before objective_mse")
        images = np.asarray(images)
        labels = np.asarray(labels)
        self._validate(images, labels, "evaluation")
        n_rows = len(images) if max_rows is None else min(max_rows, len(images))
        seed = self.random_state + 3 if random_state is None else int(random_state)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        total = 0.0
        network.eval()
        with torch.no_grad():
            for start in range(0, n_rows, self.batch_size):
                stop = min(start + self.batch_size, n_rows)
                target = self._scale(torch.from_numpy(images[start:stop])).unsqueeze(1)
                labels_batch = torch.from_numpy(
                    labels[start:stop].astype(np.int64, copy=False)
                )
                noise = torch.randn(target.shape, generator=generator)
                time = torch.rand((len(target), 1), generator=generator)
                target = target.to(self.device_)
                labels_batch = labels_batch.to(self.device_)
                noise = noise.to(self.device_)
                time = time.to(self.device_)
                state = (1.0 - time[:, :, None, None]) * noise + time[
                    :, :, None, None
                ] * target
                loss = F.mse_loss(
                    network(time, state, labels_batch), target - noise, reduction="sum"
                )
                total += float(loss.cpu())
        return total / (n_rows * 28 * 28)

    def sample(
        self,
        labels: np.ndarray,
        n_steps: int = 20,
        random_state: int | None = None,
        return_path: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("fit must be called before sample")
        labels = np.asarray(labels, dtype=np.int64)
        if labels.ndim != 1 or len(labels) == 0 or labels.min() < 0 or labels.max() > 9:
            raise ValueError("labels must be a non-empty vector in {0,...,9}")
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        seed = self.random_state + 4 if random_state is None else int(random_state)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        state = torch.randn((len(labels), 1, 28, 28), generator=generator).to(
            self.device_
        )
        label_tensor = torch.from_numpy(labels).to(self.device_)
        step_size = 1.0 / n_steps
        path = [state.detach().cpu().numpy().copy()]
        self.ema_network_.eval()
        with torch.no_grad():
            for step in range(n_steps):
                time = torch.full(
                    (len(labels), 1), step * step_size, device=self.device_
                )
                velocity = self.ema_network_(time, state, label_tensor)
                midpoint = state + 0.5 * step_size * velocity
                midpoint_time = time + 0.5 * step_size
                state = state + step_size * self.ema_network_(
                    midpoint_time, midpoint, label_tensor
                )
                state.clamp_(-3.0, 3.0)
                path.append(state.detach().cpu().numpy().copy())
        images = state.clamp(-1.0, 1.0).add(1.0).div(2.0).cpu().numpy()[:, 0]
        if return_path:
            scaled_path = np.clip((np.stack(path)[:, :, 0] + 1.0) / 2.0, 0.0, 1.0)
            return images.astype(float, copy=False), scaled_path
        return images.astype(float, copy=False)

    def state_dict(self) -> dict[str, object]:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("fit must be called before state_dict")
        return {
            "network": self.network_.state_dict(),
            "ema_network": self.ema_network_.state_dict(),
            "config": {
                "hidden_size": self.hidden_size,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "ema_decay": self.ema_decay,
                "random_state": self.random_state,
                "device": self.device,
            },
            "loss_curve": self.loss_curve_,
            "validation_loss_curve": self.validation_loss_curve_,
            "parameter_count": self.parameter_count_,
            "n_train_rows": self.n_train_rows_,
        }

    @classmethod
    def from_state_dict(
        cls, payload: dict[str, object], device: str = "auto"
    ) -> "ConditionalMNISTFlowMatcher":
        config = dict(payload["config"])
        config["device"] = device
        model = cls(**config)
        model.device_ = resolve_torch_device(device)
        model.network_ = MNISTVelocityCNN(model.hidden_size).to(model.device_)
        model.ema_network_ = MNISTVelocityCNN(model.hidden_size).to(model.device_)
        model.network_.load_state_dict(payload["network"])
        model.ema_network_.load_state_dict(payload["ema_network"])
        model.ema_network_.eval()
        for parameter in model.ema_network_.parameters():
            parameter.requires_grad_(False)
        model.loss_curve_ = [float(value) for value in payload["loss_curve"]]
        model.validation_loss_curve_ = [
            float(value) for value in payload["validation_loss_curve"]
        ]
        model.parameter_count_ = int(payload["parameter_count"])
        model.n_train_rows_ = int(payload["n_train_rows"])
        model.is_fitted_ = True
        return model
