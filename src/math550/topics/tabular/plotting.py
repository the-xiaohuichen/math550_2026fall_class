"""Publication-ready plots for the head-to-head benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "scratch": "#2F5597",
    "library": "#70AD47",
    "accent": "#C55A11",
    "neutral": "#666666",
}


def plot_head_to_head_auc(pair_table: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    minimum = min(pair_table["library_roc_auc"].min(), pair_table["scratch_roc_auc"].min())
    lower = max(0.85, float(minimum) - 0.015)
    ax.plot([lower, 1.0], [lower, 1.0], "--", color=COLORS["neutral"], label="Exact parity")
    ax.scatter(
        pair_table["library_roc_auc"],
        pair_table["scratch_roc_auc"],
        s=70,
        color=COLORS["scratch"],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    label_offsets = {
        "Logistic none": (5, 5),
        "Logistic L1": (5, -9),
        "Logistic L2": (5, -13),
        "Random forest": (-62, 10),
        "AdaBoost": (-8, 20),
        "Gradient boosting": (7, -18),
        "XGBoost style": (-68, 5),
        "LightGBM style": (5, 11),
    }
    for _, row in pair_table.iterrows():
        offset = label_offsets.get(row["family"], (5, 5))
        ax.annotate(
            row["family"],
            (row["library_roc_auc"], row["scratch_roc_auc"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
        )
    ax.set_xlim(lower, 1.002)
    ax.set_ylim(lower, 1.002)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Library implementation: test ROC AUC")
    ax.set_ylabel("Scratch implementation: test ROC AUC")
    ax.set_title("Head-to-head predictive ranking")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "head_to_head_auc.png", dpi=220)
    plt.close(fig)


def plot_auc_gaps(pair_table: pd.DataFrame, output_dir: Path) -> None:
    ordered = pair_table.sort_values("auc_gap")
    left = ordered["auc_gap"] - ordered["auc_gap_ci_low"]
    right = ordered["auc_gap_ci_high"] - ordered["auc_gap"]
    colors = [
        COLORS["scratch"] if status == "consistent" else COLORS["accent"]
        for status in ordered["consistency"]
    ]
    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    ax.errorbar(
        ordered["auc_gap"],
        ordered["family"],
        xerr=np.vstack([left, right]),
        fmt="none",
        ecolor=COLORS["neutral"],
        capsize=3,
        linewidth=1.3,
    )
    ax.scatter(ordered["auc_gap"], ordered["family"], color=colors, s=55, zorder=3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Test ROC AUC gap: scratch minus library (95% paired-bootstrap CI)")
    ax.set_title("Implementation gap with paired uncertainty")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "paired_auc_gaps.png", dpi=220)
    plt.close(fig)


def plot_probability_agreement(
    y_true: np.ndarray,
    predictions: pd.DataFrame,
    pair_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    families = pair_table["family"].tolist()
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True, sharey=True)
    for ax, family in zip(axes.ravel(), families):
        safe = family.lower().replace(" ", "_").replace("/", "_")
        scratch = predictions[f"{safe}_scratch"].to_numpy()
        library = predictions[f"{safe}_library"].to_numpy()
        ax.scatter(
            library,
            scratch,
            c=np.where(y_true == 1, COLORS["accent"], COLORS["scratch"]),
            s=15,
            alpha=0.7,
            linewidth=0,
        )
        ax.plot([0, 1], [0, 1], "--", color=COLORS["neutral"], linewidth=0.8)
        ax.set_title(family, fontsize=9)
        ax.grid(alpha=0.15)
    fig.supxlabel("Library predicted probability")
    fig.supylabel("Scratch predicted probability")
    fig.suptitle("Observation-level probability agreement", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "probability_agreement.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_logistic_optimization(
    loss_curves: dict[str, list[float]], output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for label, curve in loss_curves.items():
        values = np.asarray(curve)
        iterations = np.arange(len(values))
        keep = np.unique(
            np.r_[0, np.geomspace(1, max(1, len(values) - 1), 250).astype(int)]
        )
        keep = keep[keep < len(values)]
        ax.plot(iterations[keep], values[keep], label=label, linewidth=1.8)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Optimization iteration (symlog scale)")
    ax.set_ylabel("Penalized training objective")
    ax.set_title("Scratch logistic-regression convergence")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "logistic_optimization.png", dpi=220)
    plt.close(fig)


def plot_regularization_effect(path_table: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for penalty, group in path_table.groupby("penalty"):
        axes[0].plot(
            group["reg_strength"],
            group["validation_log_loss"],
            marker="o",
            label=penalty,
        )
        axes[1].plot(
            group["reg_strength"],
            group["coefficient_l1_norm"],
            marker="o",
            label=f"{penalty}: L1 norm",
        )
    l1 = path_table[path_table["penalty"] == "L1"]
    twin = axes[1].twinx()
    twin.plot(
        l1["reg_strength"],
        l1["nonzero_coefficients"],
        marker="s",
        color=COLORS["accent"],
        linestyle="--",
        label="L1: nonzero count",
    )
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.2)
        ax.set_xlabel("Regularization strength lambda")
    axes[0].set_ylabel("Inner-validation log loss")
    axes[0].set_title("Bias-variance trade-off")
    axes[0].legend()
    axes[1].set_ylabel("Coefficient L1 norm")
    twin.set_ylabel("Number of nonzero L1 coefficients")
    axes[1].set_title("Shrinkage and sparsity")
    handles, labels = axes[1].get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axes[1].legend(handles + twin_handles, labels + twin_labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "regularization_effect.png", dpi=220)
    plt.close(fig)
