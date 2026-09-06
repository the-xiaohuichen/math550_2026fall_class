"""Immutable configuration and matched regressors for the generative benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from math550.topics.generative import SUPPORTED_TORCH_DEVICES, TorchVectorRegressor


@dataclass(frozen=True)
class ExperimentConfig:
    output_root: Path = Path("output")
    random_state: int = 550
    test_fraction: float = 0.20
    n_components: int = 8
    hidden_sizes: tuple[int, int] = (64, 64)
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    training_pairs: int = 12_000
    validation_pairs: int = 2_000
    diffusion_steps: int = 50
    ode_steps: int = 30
    evaluation_replicates: int = 8
    generated_per_replicate: int = 500
    bootstrap_samples: int = 2_000
    device: str = "auto"
    quick: bool = False

    @property
    def analysis_dir(self) -> Path:
        return self.output_root / "generative" / "analysis"

    @property
    def figure_dir(self) -> Path:
        return self.output_root / "generative" / "figures"

    @property
    def pdf_dir(self) -> Path:
        return self.output_root / "pdf"

    @property
    def effective_epochs(self) -> int:
        return 24 if self.quick else self.max_epochs

    @property
    def effective_training_pairs(self) -> int:
        return 3_000 if self.quick else self.training_pairs

    @property
    def effective_validation_pairs(self) -> int:
        return 600 if self.quick else self.validation_pairs

    @property
    def effective_replicates(self) -> int:
        return 3 if self.quick else self.evaluation_replicates

    @property
    def effective_generated(self) -> int:
        return 240 if self.quick else self.generated_per_replicate

    def validate(self) -> None:
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must lie in (0, 1)")
        if not 2 <= self.n_components <= 64:
            raise ValueError("n_components must lie between 2 and 64")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if self.device not in SUPPORTED_TORCH_DEVICES:
            choices = ", ".join(SUPPORTED_TORCH_DEVICES)
            raise ValueError(f"device must be one of: {choices}")
        positive = [
            self.learning_rate,
            self.batch_size,
            self.max_epochs,
            self.training_pairs,
            self.validation_pairs,
            self.diffusion_steps,
            self.ode_steps,
            self.evaluation_replicates,
            self.generated_per_replicate,
        ]
        if min(positive) <= 0:
            raise ValueError("training and evaluation settings must be positive")


def build_regressor_ablation(
    config: ExperimentConfig, seed_offset: int = 0
) -> dict[str, TorchVectorRegressor]:
    """Build a PyTorch MLP and a matched linear-capacity ablation."""

    seed = config.random_state + seed_offset
    common = dict(
        hidden_sizes=config.hidden_sizes,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        max_epochs=config.effective_epochs,
        weight_decay=0.0,
        random_state=seed,
        device=config.device,
    )
    return {
        "mlp": TorchVectorRegressor(model_kind="mlp", activation="tanh", **common),
        "linear ablation": TorchVectorRegressor(model_kind="linear", **common),
    }
