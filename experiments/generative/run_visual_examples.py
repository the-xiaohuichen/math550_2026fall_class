"""Build flow-matching and VAE demonstrations on two moons and original MNIST."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import torch

from math550.topics.generative import (
    ConditionalMNISTFlowMatcher,
    ConditionalMNISTVAE,
    GaussianVAE,
    SUPPORTED_TORCH_DEVICES,
    TwoMoonsFlowMatcher,
    load_mnist,
    rbf_mmd_squared,
)


@dataclass(frozen=True)
class VisualExampleConfig:
    output_root: Path = Path("output")
    mnist_cache: Path = Path(".cache/mnist")
    device: str = "auto"
    random_state: int = 550
    quick: bool = False
    force: bool = False
    mnist_epochs: int | None = None
    mnist_max_rows: int | None = None

    @property
    def analysis_dir(self) -> Path:
        return self.output_root / "generative" / "analysis"

    @property
    def figure_dir(self) -> Path:
        return self.output_root / "generative" / "figures"

    @property
    def model_dir(self) -> Path:
        return self.output_root / "generative" / "models"

    @property
    def effective_mnist_epochs(self) -> int:
        if self.mnist_epochs is not None:
            return self.mnist_epochs
        return 1 if self.quick else 8

    @property
    def effective_mnist_vae_epochs(self) -> int:
        if self.mnist_epochs is not None:
            return self.mnist_epochs
        return 1 if self.quick else 8

    @property
    def effective_mnist_rows(self) -> int | None:
        if self.mnist_max_rows is not None:
            return self.mnist_max_rows
        return 4_000 if self.quick else None

    @property
    def effective_moons_updates(self) -> int:
        return 1_000 if self.quick else 10_000

    @property
    def effective_moons_vae_epochs(self) -> int:
        return 250 if self.quick else 2_500

    @property
    def effective_hidden_size(self) -> int:
        return 8 if self.quick else 16


def _plot_two_moons_path(
    target: np.ndarray, path: np.ndarray, output_path: Path
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(10.2, 8.4), constrained_layout=True)
    for step, axis in enumerate(axes.flat):
        axis.scatter(
            target[:, 0],
            target[:, 1],
            s=8,
            color="#B8C0CA",
            alpha=0.38,
            edgecolors="none",
            label="target" if step == 0 else None,
        )
        axis.scatter(
            path[step, :, 0],
            path[step, :, 1],
            s=11,
            color="#2F5A9E",
            alpha=0.68,
            edgecolors="none",
            label="transported particles" if step == 0 else None,
        )
        axis.set_title(f"t = {step / 8:.3g}", loc="left", fontweight="bold")
        axis.set_xlim(-3.2, 3.2)
        axis.set_ylim(-2.2, 2.2)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[["top", "right", "bottom", "left"]].set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8, loc="lower left")
    figure.suptitle(
        "Eight midpoint steps transport Gaussian noise into two moons",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_two_moons_vae(
    target: np.ndarray,
    posterior_mean: np.ndarray,
    prior_latent: np.ndarray,
    generated: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(13.0, 3.35), constrained_layout=True)
    panels = [
        ("Observed data", target, "#2F5A9E"),
        ("Encoded means", posterior_mean, "#D9782D"),
        ("Prior draws", prior_latent, "#7A8491"),
        ("Decoded samples", generated, "#7A4EAB"),
    ]
    for axis, (title, values, color) in zip(axes, panels):
        axis.scatter(values[:, 0], values[:, 1], s=12, alpha=0.65, color=color, edgecolors="none")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[["top", "right", "bottom", "left"]].set_visible(False)
    axes[0].set_xlim(-1.4, 2.4)
    axes[0].set_ylim(-0.9, 1.4)
    axes[3].set_xlim(-1.4, 2.4)
    axes[3].set_ylim(-0.9, 1.4)
    figure.suptitle(
        "The VAE regularizes an encoder distribution, then generates in one decoder pass",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_mnist_grid(
    images: np.ndarray, labels: np.ndarray, output_path: Path
) -> None:
    rows = len(images) // 10
    figure, axes = plt.subplots(rows, 10, figsize=(12.0, 1.22 * rows))
    for index, (image, label) in enumerate(zip(images, labels)):
        row, column = divmod(index, 10)
        axis = axes[row, column]
        axis.imshow(image, cmap="gray_r", vmin=0.0, vmax=1.0)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        if row == 0:
            axis.set_title(str(int(label)), fontsize=10, fontweight="bold")
    figure.suptitle(
        "Class-conditional 28×28 MNIST samples (column = requested digit)",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.01, 0.0, 1.0, 0.95), w_pad=0.05, h_pad=0.05)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_mnist_path(
    path: np.ndarray, labels: np.ndarray, output_path: Path
) -> None:
    selected_steps = [0, 5, 10, 15, 20]
    selected_rows = [0, 2, 5, 8]
    figure, axes = plt.subplots(4, 5, figsize=(9.0, 7.3), constrained_layout=True)
    for row, sample_index in enumerate(selected_rows):
        for column, step in enumerate(selected_steps):
            axis = axes[row, column]
            axis.imshow(path[step, sample_index], cmap="gray_r", vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(f"t={step / 20:.2f}", fontweight="bold")
            if column == 0:
                axis.set_ylabel(
                    f"label {int(labels[sample_index])}",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=20,
                    fontweight="bold",
                )
    figure.suptitle(
        "The learned ODE progressively resolves noise into controlled digits",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_mnist_training(
    training: list[float], validation: list[float], output_path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    epochs = np.arange(1, len(training) + 1)
    axis.plot(epochs, training, marker="o", linewidth=2.2, label="training")
    if validation:
        axis.plot(epochs, validation, marker="s", linewidth=2.2, label="test-set probe")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Flow-matching velocity MSE per pixel")
    axis.set_title("The PyTorch objective declines on train and held-out pairs", loc="left", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9DCE1", linewidth=0.7)
    axis.legend(frameon=False)
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_mnist_vae_training(
    objective: list[float],
    reconstruction: list[float],
    kl: list[float],
    validation: list[float],
    output_path: Path,
) -> None:
    epochs = np.arange(1, len(objective) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].plot(epochs, objective, marker="o", linewidth=2.1, label="training negative ELBO")
    if validation:
        axes[0].plot(epochs, validation, marker="s", linewidth=2.1, label="held-out negative ELBO")
    axes[0].set_ylabel("Nats per image")
    axes[0].set_title("The variational objective decreases", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(epochs, reconstruction, marker="o", linewidth=2.1, label="reconstruction BCE")
    axes[1].plot(epochs, kl, marker="s", linewidth=2.1, label="posterior KL")
    axes[1].set_ylabel("Nats per image")
    axes[1].set_title("Reconstruction and regularization stay visible", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9DCE1", linewidth=0.7)
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_mnist_vae_reconstructions(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, len(originals), figsize=(12.0, 2.8), constrained_layout=True)
    for column, (original, reconstruction, label) in enumerate(
        zip(originals, reconstructions, labels)
    ):
        for row, image in enumerate([original, reconstruction]):
            axes[row, column].imshow(image, cmap="gray_r", vmin=0.0, vmax=1.0)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].spines[["top", "right", "bottom", "left"]].set_visible(False)
        axes[0, column].set_title(str(int(label)), fontsize=9, fontweight="bold")
    axes[0, 0].set_ylabel("held-out", rotation=0, ha="right", va="center", labelpad=12)
    axes[1, 0].set_ylabel("reconstruction", rotation=0, ha="right", va="center", labelpad=12)
    figure.suptitle("Posterior-mean reconstructions expose what the bottleneck preserves", fontsize=13, fontweight="bold")
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _two_moons_example(config: VisualExampleConfig) -> dict[str, object]:
    target, _ = make_moons(
        n_samples=256, noise=0.05, random_state=config.random_state
    )
    start = time.perf_counter()
    model = TwoMoonsFlowMatcher(
        hidden_size=64,
        learning_rate=1e-2,
        updates=config.effective_moons_updates,
        random_state=config.random_state,
        device=config.device,
    ).fit(target)
    fit_seconds = time.perf_counter() - start
    display, path = model.sample(
        300, n_steps=8, random_state=config.random_state + 10, return_path=True
    )
    generated = model.sample(
        1_000, n_steps=8, random_state=config.random_state + 11
    )
    reference, _ = make_moons(
        n_samples=1_000, noise=0.05, random_state=config.random_state + 12
    )
    heldout, _ = make_moons(
        n_samples=1_000, noise=0.05, random_state=config.random_state + 13
    )
    nearest = NearestNeighbors(n_neighbors=1).fit(reference)
    heldout_distance = nearest.kneighbors(heldout)[0][:, 0]
    generated_distance = nearest.kneighbors(generated)[0][:, 0]
    support_threshold = float(np.quantile(heldout_distance, 0.95))
    _plot_two_moons_path(
        target, path, config.figure_dir / "two_moons_flow_path.png"
    )
    return {
        "dataset": "sklearn.datasets.make_moons",
        "n_training_points": 256,
        "noise": 0.05,
        "source_distribution": "standard two-dimensional Gaussian",
        "coupling": "independent pairing",
        "interpolation": "x_t=(1-t)x_0+t x_1; target velocity=x_1-x_0",
        "network": "PyTorch MLP with three width-64 ELU hidden layers",
        "optimizer": "torch.optim.Adam",
        "learning_rate": 0.01,
        "updates": config.effective_moons_updates,
        "sampler": "midpoint ODE rule",
        "sampling_steps": 8,
        "parameter_count": model.parameter_count_,
        "final_recorded_loss": model.loss_curve_[-1],
        "fit_seconds": fit_seconds,
        "rbf_mmd_squared": rbf_mmd_squared(reference, generated),
        "median_nearest_reference": float(np.median(generated_distance)),
        "heldout_median_nearest_reference": float(np.median(heldout_distance)),
        "support_threshold_95pct_heldout": support_threshold,
        "generated_support_fraction": float(
            np.mean(generated_distance <= support_threshold)
        ),
        "requested_device": config.device,
        "resolved_device": str(model.device_),
        "figure": "two_moons_flow_path.png",
        "claim_scope": (
            "fixed-seed visual and geometric pipeline evidence; not a broad model comparison"
        ),
    }


def _two_moons_vae_example(config: VisualExampleConfig) -> dict[str, object]:
    target, _ = make_moons(
        n_samples=256, noise=0.05, random_state=config.random_state
    )
    reference, _ = make_moons(
        n_samples=1_000, noise=0.05, random_state=config.random_state + 12
    )
    heldout, _ = make_moons(
        n_samples=1_000, noise=0.05, random_state=config.random_state + 13
    )
    start = time.perf_counter()
    model = GaussianVAE(
        latent_dim=2,
        hidden_sizes=(64, 64),
        observation_sigma=0.05,
        beta=1.0,
        learning_rate=1e-3,
        batch_size=256,
        max_epochs=config.effective_moons_vae_epochs,
        random_state=config.random_state + 40,
        device=config.device,
    ).fit(target, heldout)
    fit_seconds = time.perf_counter() - start
    generated, prior_latent = model.sample(
        1_000,
        random_state=config.random_state + 41,
        sample_observation=True,
        return_latent=True,
    )
    posterior_mean, posterior_std = model.encode(target)
    nearest = NearestNeighbors(n_neighbors=1).fit(reference)
    heldout_distance = nearest.kneighbors(heldout)[0][:, 0]
    generated_distance = nearest.kneighbors(generated)[0][:, 0]
    support_threshold = float(np.quantile(heldout_distance, 0.95))
    _plot_two_moons_vae(
        target,
        posterior_mean,
        prior_latent,
        generated,
        config.figure_dir / "two_moons_vae.png",
    )
    evaluation = model.evaluate(heldout)
    return {
        "dataset": "sklearn.datasets.make_moons",
        "n_training_points": 256,
        "noise": 0.05,
        "model": "PyTorch Gaussian VAE with two width-64 ELU encoder and decoder layers",
        "latent_dimension": 2,
        "decoder_likelihood": "isotropic Gaussian with fixed sigma=0.05",
        "posterior": "diagonal Gaussian",
        "objective": "negative ELBO up to the fixed Gaussian normalizing constant",
        "beta": 1.0,
        "optimizer": "torch.optim.Adam",
        "learning_rate": 0.001,
        "epochs": len(model.loss_curve_),
        "parameter_count": model.parameter_count_,
        "final_training_objective": model.loss_curve_[-1],
        "heldout_objective": evaluation["objective"],
        "heldout_reconstruction_mse": evaluation["reconstruction_mse"],
        "heldout_mean_kl": evaluation["kl"],
        "mean_posterior_std": float(posterior_std.mean()),
        "fit_seconds": fit_seconds,
        "sampling_network_evaluations": 1,
        "rbf_mmd_squared": rbf_mmd_squared(reference, generated),
        "median_nearest_reference": float(np.median(generated_distance)),
        "heldout_median_nearest_reference": float(np.median(heldout_distance)),
        "support_threshold_95pct_heldout": support_threshold,
        "generated_support_fraction": float(
            np.mean(generated_distance <= support_threshold)
        ),
        "requested_device": config.device,
        "resolved_device": str(model.device_),
        "figure": "two_moons_vae.png",
        "claim_scope": (
            "fixed-seed latent-variable and one-pass sampling evidence; not a broad VAE comparison"
        ),
    }


def _mnist_features(images: np.ndarray) -> np.ndarray:
    values = images.astype(np.float32)
    if values.max() > 1.0:
        values = values / 255.0
    return values.reshape(len(values), 14, 2, 14, 2).mean(axis=(2, 4)).reshape(len(values), -1)


def _mnist_probe_metrics(
    data: object,
    generated: np.ndarray,
    labels: np.ndarray,
    random_state: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(random_state)
    classifier_rows = min(20_000, len(data.train_images))
    classifier_indices = rng.choice(
        len(data.train_images), size=classifier_rows, replace=False
    )
    classifier = KNeighborsClassifier(n_neighbors=3, weights="distance").fit(
        _mnist_features(data.train_images[classifier_indices]),
        data.train_labels[classifier_indices],
    )
    test_rows = min(2_000, len(data.test_images))
    probe_accuracy = float(
        np.mean(
            classifier.predict(_mnist_features(data.test_images[:test_rows]))
            == data.test_labels[:test_rows]
        )
    )
    predicted = classifier.predict(_mnist_features(generated))
    nearest_rows = min(10_000, classifier_rows)
    nearest = NearestNeighbors(n_neighbors=1).fit(
        _mnist_features(data.train_images[classifier_indices[:nearest_rows]])
    )
    generated_nearest = nearest.kneighbors(_mnist_features(generated))[0][:, 0]
    test_nearest = nearest.kneighbors(_mnist_features(data.test_images[:500]))[0][:, 0]
    return {
        "knn_probe_training_rows": classifier_rows,
        "knn_probe_test_rows": test_rows,
        "knn_probe_test_accuracy": probe_accuracy,
        "requested_label_accuracy_under_probe": float(np.mean(predicted == labels)),
        "generated_median_nearest_train_14x14": float(np.median(generated_nearest)),
        "heldout_median_nearest_train_14x14": float(np.median(test_nearest)),
        "mean_within_label_pixel_std": float(
            np.mean(
                [generated[labels == label].std(axis=0).mean() for label in range(10)]
            )
        ),
    }


def _mnist_example(config: VisualExampleConfig) -> dict[str, object]:
    data = load_mnist(config.mnist_cache, download=True)
    checkpoint = config.model_dir / (
        "mnist_28_flow_quick.pt" if config.quick else "mnist_28_flow.pt"
    )
    reused = checkpoint.exists() and not config.force
    start = time.perf_counter()
    if reused:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = ConditionalMNISTFlowMatcher.from_state_dict(payload, device=config.device)
    else:
        model = ConditionalMNISTFlowMatcher(
            hidden_size=config.effective_hidden_size,
            learning_rate=1e-3,
            batch_size=256,
            max_epochs=config.effective_mnist_epochs,
            ema_decay=0.995,
            random_state=config.random_state,
            device=config.device,
        ).fit(
            data.train_images,
            data.train_labels,
            validation_images=data.test_images,
            validation_labels=data.test_labels,
            max_train_rows=config.effective_mnist_rows,
        )
        torch.save(model.state_dict(), checkpoint)
    fit_seconds = time.perf_counter() - start

    labels = np.tile(np.arange(10, dtype=np.int64), 8)
    generated, path = model.sample(
        labels, n_steps=20, random_state=config.random_state + 20, return_path=True
    )
    _plot_mnist_grid(
        generated, labels, config.figure_dir / "mnist_28_generated.png"
    )
    _plot_mnist_path(
        path, labels, config.figure_dir / "mnist_28_flow_progression.png"
    )
    _plot_mnist_training(
        model.loss_curve_,
        model.validation_loss_curve_,
        config.figure_dir / "mnist_28_training_curve.png",
    )
    np.savez_compressed(
        config.analysis_dir / "mnist_28_generated_samples.npz",
        images=generated.astype(np.float32),
        requested_labels=labels,
        seed=config.random_state + 20,
    )

    probe = _mnist_probe_metrics(
        data, generated, labels, config.random_state + 21
    )

    return {
        "dataset": data.metadata.name,
        "dataset_source": data.metadata.source_url,
        "dataset_license": data.metadata.license,
        "mirror": "torchvision MNIST mirror at ossci-datasets.s3.amazonaws.com",
        "file_md5": data.file_md5,
        "n_train": len(data.train_images),
        "n_test": len(data.test_images),
        "image_shape": list(data.metadata.image_shape),
        "n_training_rows_used": model.n_train_rows_,
        "model": "class-conditional PyTorch flow-matching U-Net-style CNN",
        "conditioning": "one-hot requested digit label",
        "base_channels": model.hidden_size,
        "parameter_count": model.parameter_count_,
        "epochs": len(model.loss_curve_),
        "training_loss_curve": model.loss_curve_,
        "validation_loss_curve": model.validation_loss_curve_,
        "final_training_velocity_mse": model.loss_curve_[-1],
        "final_test_pair_velocity_mse": model.objective_mse(
            data.test_images,
            data.test_labels,
            max_rows=2_000,
            random_state=config.random_state + 30,
        ),
        "optimizer": "torch.optim.AdamW",
        "sampler": "midpoint ODE rule",
        "sampling_steps": 20,
        "generated_samples": len(generated),
        **probe,
        "fit_or_load_seconds": fit_seconds,
        "checkpoint_reused": reused,
        "requested_device": config.device,
        "resolved_device": str(model.device_),
        "figures": [
            "mnist_28_generated.png",
            "mnist_28_flow_progression.png",
            "mnist_28_training_curve.png",
        ],
        "claim_scope": (
            "class-conditional visual and pipeline evidence; k-NN agreement is not human judgment, FID, or a privacy audit"
        ),
    }


def _mnist_vae_example(config: VisualExampleConfig) -> dict[str, object]:
    data = load_mnist(config.mnist_cache, download=True)
    checkpoint = config.model_dir / (
        "mnist_28_vae_quick.pt" if config.quick else "mnist_28_vae.pt"
    )
    reused = checkpoint.exists() and not config.force
    start = time.perf_counter()
    if reused:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = ConditionalMNISTVAE.from_state_dict(payload, device=config.device)
    else:
        model = ConditionalMNISTVAE(
            latent_dim=16,
            hidden_size=config.effective_hidden_size,
            beta=1.0,
            learning_rate=1e-3,
            batch_size=256,
            max_epochs=config.effective_mnist_vae_epochs,
            random_state=config.random_state + 100,
            device=config.device,
        ).fit(
            data.train_images,
            data.train_labels,
            validation_images=data.test_images,
            validation_labels=data.test_labels,
            max_train_rows=config.effective_mnist_rows,
        )
        torch.save(model.state_dict(), checkpoint)
    fit_seconds = time.perf_counter() - start

    labels = np.tile(np.arange(10, dtype=np.int64), 8)
    generated = model.sample(labels, random_state=config.random_state + 120)
    reconstruction_indices = np.array(
        [np.flatnonzero(data.test_labels == label)[0] for label in range(10)]
    )
    reconstruction_images = data.test_images[reconstruction_indices]
    reconstruction_labels = data.test_labels[reconstruction_indices]
    reconstructions = model.reconstruct(
        reconstruction_images, reconstruction_labels
    )
    _plot_mnist_grid(
        generated, labels, config.figure_dir / "mnist_28_vae_generated.png"
    )
    _plot_mnist_vae_reconstructions(
        reconstruction_images.astype(float) / 255.0,
        reconstructions,
        reconstruction_labels,
        config.figure_dir / "mnist_28_vae_reconstruction.png",
    )
    _plot_mnist_vae_training(
        model.loss_curve_,
        model.reconstruction_curve_,
        model.kl_curve_,
        model.validation_loss_curve_,
        config.figure_dir / "mnist_28_vae_training_curve.png",
    )
    np.savez_compressed(
        config.analysis_dir / "mnist_28_vae_generated_samples.npz",
        images=generated.astype(np.float32),
        requested_labels=labels,
        seed=config.random_state + 120,
    )
    heldout = model.evaluate(
        data.test_images, data.test_labels, max_rows=2_000
    )
    probe = _mnist_probe_metrics(
        data, generated, labels, config.random_state + 121
    )
    return {
        "dataset": data.metadata.name,
        "dataset_source": data.metadata.source_url,
        "dataset_license": data.metadata.license,
        "mirror": "torchvision MNIST mirror at ossci-datasets.s3.amazonaws.com",
        "file_md5": data.file_md5,
        "n_train": len(data.train_images),
        "n_test": len(data.test_images),
        "image_shape": list(data.metadata.image_shape),
        "n_training_rows_used": model.n_train_rows_,
        "model": "class-conditional PyTorch convolutional VAE",
        "conditioning": "one-hot digit label in encoder and decoder",
        "decoder_likelihood": "independent Bernoulli pixels",
        "posterior": "16-dimensional diagonal Gaussian",
        "objective": "negative ELBO = pixel BCE + KL(q_phi(z|x,y)||N(0,I))",
        "beta": model.beta,
        "base_channels": model.hidden_size,
        "latent_dimension": model.latent_dim,
        "parameter_count": model.parameter_count_,
        "epochs": len(model.loss_curve_),
        "training_loss_curve": model.loss_curve_,
        "training_reconstruction_curve": model.reconstruction_curve_,
        "training_kl_curve": model.kl_curve_,
        "validation_loss_curve": model.validation_loss_curve_,
        "final_training_negative_elbo": model.loss_curve_[-1],
        "final_training_reconstruction_bce": model.reconstruction_curve_[-1],
        "final_training_kl": model.kl_curve_[-1],
        "heldout_negative_elbo": heldout["negative_elbo"],
        "heldout_reconstruction_bce": heldout["reconstruction_bce"],
        "heldout_kl": heldout["kl"],
        "optimizer": "torch.optim.AdamW",
        "sampling_network_evaluations": 1,
        "generated_samples": len(generated),
        **probe,
        "fit_or_load_seconds": fit_seconds,
        "checkpoint_reused": reused,
        "requested_device": config.device,
        "resolved_device": str(model.device_),
        "figures": [
            "mnist_28_vae_generated.png",
            "mnist_28_vae_reconstruction.png",
            "mnist_28_vae_training_curve.png",
        ],
        "claim_scope": (
            "class-conditional visual, reconstruction, and ELBO evidence; probe agreement is not human judgment, FID, or a privacy audit"
        ),
    }


def run(config: VisualExampleConfig) -> Path:
    if config.device not in SUPPORTED_TORCH_DEVICES:
        raise ValueError("unsupported device policy")
    if config.effective_mnist_epochs < 1:
        raise ValueError("mnist epochs must be positive")
    for directory in [config.analysis_dir, config.figure_dir, config.model_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    print("Training the tutorial two-moons flow...")
    moons = _two_moons_example(config)
    print(
        f"two moons: MMD²={moons['rbf_mmd_squared']:.5f}, "
        f"support={moons['generated_support_fraction']:.3f}"
    )
    print("Training the two-moons VAE...")
    moons_vae = _two_moons_vae_example(config)
    print(
        f"two-moons VAE: MMD²={moons_vae['rbf_mmd_squared']:.5f}, "
        f"KL={moons_vae['heldout_mean_kl']:.3f}"
    )
    print("Training or loading the original-MNIST 28x28 flow...")
    mnist = _mnist_example(config)
    print(
        f"MNIST: final held-out objective={mnist['final_test_pair_velocity_mse']:.4f}, "
        f"label-probe agreement={mnist['requested_label_accuracy_under_probe']:.3f}"
    )
    print("Training or loading the original-MNIST 28x28 VAE...")
    mnist_vae = _mnist_vae_example(config)
    print(
        f"MNIST VAE: held-out negative ELBO={mnist_vae['heldout_negative_elbo']:.2f}, "
        f"label-probe agreement={mnist_vae['requested_label_accuracy_under_probe']:.3f}"
    )
    output = config.analysis_dir / "visual_examples.json"
    output.write_text(
        json.dumps(
            {
                "topic": "generative",
                "random_state": config.random_state,
                "quick": config.quick,
                "framework": "PyTorch",
                "torch_version": torch.__version__,
                "two_moons": moons,
                "two_moons_vae": moons_vae,
                "mnist_28": mnist,
                "mnist_28_vae": mnist_vae,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Visual-example evidence: {output.resolve()}")
    return output


def parse_config() -> VisualExampleConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--mnist-cache", type=Path, default=Path(".cache/mnist"))
    parser.add_argument("--device", choices=SUPPORTED_TORCH_DEVICES, default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mnist-epochs", type=int)
    parser.add_argument("--mnist-max-rows", type=int)
    arguments = parser.parse_args()
    return VisualExampleConfig(
        output_root=arguments.output_root,
        mnist_cache=arguments.mnist_cache,
        device=arguments.device,
        quick=arguments.quick,
        force=arguments.force,
        mnist_epochs=arguments.mnist_epochs,
        mnist_max_rows=arguments.mnist_max_rows,
    )


if __name__ == "__main__":
    run(parse_config())
