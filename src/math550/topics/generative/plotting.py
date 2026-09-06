"""Figures driven only by completed generative-model experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "mlp": "#2F5A9E",
    "linear ablation": "#D9782D",
    "baseline": "#147D64",
    "accent": "#3D8DFF",
    "muted": "#7A8491",
}


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9DCE1", linewidth=0.7, alpha=0.75)


def plot_learning_curves(loss_curves: dict[str, list[float]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)
    specifications = [
        ("Flow matching", ["mlp", "linear ablation"]),
        ("Diffusion", ["mlp", "linear ablation"]),
        ("VAE", ["mlp"]),
    ]
    for axis, (family, implementations) in zip(axes, specifications):
        for implementation in implementations:
            values = loss_curves[f"{family}|{implementation}"]
            axis.plot(
                np.arange(1, len(values) + 1),
                values,
                color=COLORS[implementation],
                linewidth=2.2,
                label=implementation.upper() if implementation == "mlp" else "Linear ablation",
            )
        axis.set_title(family, loc="left", fontweight="bold")
        axis.set_xlabel("Recorded epoch")
        axis.set_ylabel("Training objective")
        axis.set_yscale("log")
        _style_axis(axis)
        axis.legend(frameon=False)
    figure.suptitle(
        "PyTorch exposes how model capacity changes target regression",
        fontweight="bold",
    )
    path = output_dir / "learning_curves.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_metric_comparison(metrics: pd.DataFrame, output_dir: Path) -> Path:
    summary = (
        metrics.groupby(["family", "implementation"])[
            ["frechet_latent", "mmd_rbf", "precision_knn", "recall_knn"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    specifications = [
        ("frechet_latent", "Latent Fréchet distance", "lower is better"),
        ("mmd_rbf", "RBF MMD²", "lower is better"),
        ("precision_knn", "k-NN precision", "higher is better"),
        ("recall_knn", "k-NN recall", "higher is better"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = [
        "Flow\nMLP",
        "Flow\nlinear",
        "Diffusion\nMLP",
        "Diffusion\nlinear",
        "VAE\nMLP",
        "GMM\nbaseline",
    ]
    selectors = [
        ("Flow matching", "mlp"),
        ("Flow matching", "linear ablation"),
        ("Diffusion", "mlp"),
        ("Diffusion", "linear ablation"),
        ("VAE", "mlp"),
        ("Gaussian mixture", "classical baseline"),
    ]
    colors = [
        COLORS["mlp"],
        COLORS["linear ablation"],
        COLORS["mlp"],
        COLORS["linear ablation"],
        "#7A4EAB",
        COLORS["baseline"],
    ]
    for axis, (metric, title, direction) in zip(axes.flat, specifications):
        means, standard_deviations = [], []
        for family, implementation in selectors:
            row = summary[
                (summary["family"] == family)
                & (summary["implementation"] == implementation)
            ].iloc[0]
            means.append(float(row[(metric, "mean")]))
            standard_deviations.append(float(row[(metric, "std")]))
        axis.bar(
            np.arange(len(labels)),
            means,
            yerr=standard_deviations,
            color=colors,
            capsize=3,
            alpha=0.92,
        )
        axis.set_xticks(np.arange(len(labels)), labels)
        axis.set_title(f"{title}\n{direction}", loc="left", fontweight="bold")
        _style_axis(axis)
    path = output_dir / "distribution_metrics.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_sampling_tradeoff(tradeoff: pd.DataFrame, output_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    for family, color, marker in [
        ("Flow matching", COLORS["mlp"], "o"),
        ("Diffusion", COLORS["linear ablation"], "s"),
        ("VAE", "#7A4EAB", "D"),
    ]:
        table = tradeoff[tradeoff["family"] == family].sort_values("network_evaluations")
        axes[0].plot(
            table["network_evaluations"],
            table["frechet_latent"],
            marker=marker,
            linewidth=2.2,
            color=color,
            label=family,
        )
        axes[1].plot(
            table["seconds"],
            table["frechet_latent"],
            marker=marker,
            linewidth=2.2,
            color=color,
            label=family,
        )
    axes[0].set_xlabel("Network evaluations per sample")
    axes[0].set_ylabel("Latent Fréchet distance")
    axes[0].set_title("Numerical work vs. quality", loc="left", fontweight="bold")
    axes[1].set_xlabel("Wall-clock seconds for the batch")
    axes[1].set_ylabel("Latent Fréchet distance")
    axes[1].set_title("Observed CPU time vs. quality", loc="left", fontweight="bold")
    for axis in axes:
        _style_axis(axis)
        axis.legend(frameon=False)
    path = output_dir / "sampling_tradeoff.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_latent_samples(
    test_latent: np.ndarray, display_samples: dict[str, np.ndarray], output_dir: Path
) -> Path:
    panels = [("Held-out data", test_latent), *display_samples.items()]
    figure, axes = plt.subplots(3, 3, figsize=(12, 10.2), constrained_layout=True)
    for axis, (label, values) in zip(axes.flat, panels):
        axis.scatter(
            values[:, 0],
            values[:, 1],
            s=13,
            alpha=0.55,
            color=COLORS["accent"] if label == "Held-out data" else COLORS["muted"],
            edgecolors="none",
        )
        axis.set_title(label.replace("|", " - "), loc="left", fontsize=10, fontweight="bold")
        axis.set_xlabel("Whitened PC 1")
        axis.set_ylabel("Whitened PC 2")
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes.flat[len(panels) :]:
        axis.axis("off")
    path = output_dir / "latent_samples.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_sample_grids(data: object, display_samples: dict[str, np.ndarray], output_dir: Path) -> Path:
    rows = [("Held-out data", data.test_latent[:8]), *[(key, value[:8]) for key, value in display_samples.items()]]
    figure, axes = plt.subplots(len(rows), 8, figsize=(10, 1.55 * len(rows)))
    for row_index, (label, latent) in enumerate(rows):
        images = data.inverse_transform(latent)
        for column, image in enumerate(images):
            axis = axes[row_index, column]
            axis.imshow(image, cmap="gray_r", vmin=0.0, vmax=1.0, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if column == 0:
                axis.set_ylabel(
                    label.replace("|", "\n"),
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=8,
                    labelpad=8,
                )
    figure.suptitle(
        "Decoded samples (PCA bottleneck limits sharpness)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.10, 0.0, 1.0, 0.97), h_pad=0.45, w_pad=0.15)
    path = output_dir / "decoded_sample_grid.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_transport_trajectories(
    train_latent: np.ndarray, path_values: np.ndarray, output_dir: Path
) -> Path:
    figure, axis = plt.subplots(figsize=(8.5, 6.3), constrained_layout=True)
    axis.scatter(
        train_latent[:, 0],
        train_latent[:, 1],
        s=8,
        alpha=0.16,
        color="#9AA3AD",
        label="Training data",
    )
    for sample_index in range(path_values.shape[1]):
        axis.plot(
            path_values[:, sample_index, 0],
            path_values[:, sample_index, 1],
            color=COLORS["mlp"],
            alpha=0.43,
            linewidth=1.0,
        )
    axis.scatter(
        path_values[0, :, 0],
        path_values[0, :, 1],
        s=14,
        color=COLORS["linear ablation"],
        label="Noise at t=0",
    )
    axis.scatter(
        path_values[-1, :, 0],
        path_values[-1, :, 1],
        s=18,
        color=COLORS["mlp"],
        label="Generated at t=1",
    )
    axis.set_xlabel("Whitened PC 1")
    axis.set_ylabel("Whitened PC 2")
    axis.set_title("The learned vector field transports noise toward data", loc="left", fontweight="bold")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    path = output_dir / "flow_trajectories.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path
