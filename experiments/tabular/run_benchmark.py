"""Run the controlled scratch-versus-library tabular benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from math550.topics.tabular import load_wdbc
from math550.topics.tabular.evaluation import (
    binary_metrics,
    probability_agreement,
    stratified_paired_bootstrap_gap,
    validate_probability_output,
)
from math550.topics.tabular.plotting import (
    plot_auc_gaps,
    plot_head_to_head_auc,
    plot_logistic_optimization,
    plot_probability_agreement,
    plot_regularization_effect,
)
from math550.topics.tabular.reporting import (
    compile_latex_report,
    write_latex_report,
)

from .config import ExperimentConfig, PairSpec, build_pairs
from .diagnostics import (
    logistic_parameter_checks,
    model_configuration_table,
    regularization_diagnostic,
)


def safe_name(family: str) -> str:
    """Convert a model-family label into a stable output-column prefix."""

    return family.lower().replace(" ", "_").replace("/", "_")


def fit_pairs(
    pairs: list[PairSpec],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Fit every pair and return metrics, paired gaps, predictions, and fits."""

    metric_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    predictions = pd.DataFrame({"row_id": X_test.index, "actual": y_test.to_numpy()})
    fitted: dict[str, object] = {}

    for offset, pair in enumerate(pairs):
        print(f"\n{pair.family}")
        probabilities: dict[str, np.ndarray] = {}
        training_metrics: dict[str, dict[str, float]] = {}
        test_metrics: dict[str, dict[str, float]] = {}
        fit_times: dict[str, float] = {}
        for implementation, estimator in [
            ("scratch", pair.scratch),
            ("library", pair.library),
        ]:
            start = time.perf_counter()
            estimator.fit(X_train, y_train)
            fit_times[implementation] = time.perf_counter() - start
            train_probability = validate_probability_output(
                estimator.predict_proba(X_train), len(X_train)
            )
            test_probability = validate_probability_output(
                estimator.predict_proba(X_test), len(X_test)
            )
            probabilities[implementation] = test_probability
            training_metrics[implementation] = binary_metrics(
                y_train.to_numpy(), train_probability
            )
            test_metrics[implementation] = binary_metrics(
                y_test.to_numpy(), test_probability
            )
            metric_rows.append(
                {
                    "family": pair.family,
                    "implementation": implementation,
                    "note": pair.note,
                    "fit_seconds": fit_times[implementation],
                    **{
                        f"train_{name}": value
                        for name, value in training_metrics[implementation].items()
                    },
                    **{
                        f"test_{name}": value
                        for name, value in test_metrics[implementation].items()
                    },
                }
            )
            fitted[f"{pair.family}|{implementation}"] = estimator
            print(
                f"  {implementation:7s} "
                f"train/test AUC={training_metrics[implementation]['roc_auc']:.4f}/"
                f"{test_metrics[implementation]['roc_auc']:.4f} "
                f"test accuracy={test_metrics[implementation]['accuracy']:.4f} "
                f"test logloss={test_metrics[implementation]['log_loss']:.4f} "
                f"fit={fit_times[implementation]:.2f}s"
            )

        gap, low, high = stratified_paired_bootstrap_gap(
            y_test.to_numpy(),
            probabilities["scratch"],
            probabilities["library"],
            n_bootstrap=config.n_bootstrap,
            random_state=config.random_state + offset,
        )
        agreement = probability_agreement(
            probabilities["scratch"], probabilities["library"]
        )
        scratch_metrics = test_metrics["scratch"]
        library_metrics = test_metrics["library"]
        consistency = (
            "consistent" if (low <= 0 <= high or abs(gap) <= 0.02) else "review"
        )
        pair_rows.append(
            {
                "family": pair.family,
                "scratch_train_roc_auc": training_metrics["scratch"]["roc_auc"],
                "library_train_roc_auc": training_metrics["library"]["roc_auc"],
                "train_auc_gap": (
                    training_metrics["scratch"]["roc_auc"]
                    - training_metrics["library"]["roc_auc"]
                ),
                "scratch_roc_auc": scratch_metrics["roc_auc"],
                "library_roc_auc": library_metrics["roc_auc"],
                "auc_gap": gap,
                "auc_gap_ci_low": low,
                "auc_gap_ci_high": high,
                "scratch_accuracy": scratch_metrics["accuracy"],
                "library_accuracy": library_metrics["accuracy"],
                "scratch_log_loss": scratch_metrics["log_loss"],
                "library_log_loss": library_metrics["log_loss"],
                "scratch_fit_seconds": fit_times["scratch"],
                "library_fit_seconds": fit_times["library"],
                "fit_time_ratio": fit_times["scratch"] / fit_times["library"],
                **agreement,
                "consistency": consistency,
            }
        )
        prefix = safe_name(pair.family)
        predictions[f"{prefix}_scratch"] = probabilities["scratch"]
        predictions[f"{prefix}_library"] = probabilities["library"]
        print(
            f"  paired AUC gap={gap:+.4f} [{low:+.4f}, {high:+.4f}] "
            f"-> {consistency}"
        )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(pair_rows),
        predictions,
        fitted,
    )


def write_outputs(
    config: ExperimentConfig,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    parameter_table: pd.DataFrame,
    configuration_table: pd.DataFrame,
    regularization_table: pd.DataFrame,
    run_metadata: dict[str, object],
) -> None:
    """Write topic-scoped machine-readable outputs."""

    for table in [metrics, comparison, parameter_table, regularization_table]:
        numeric = table.select_dtypes(include="number").columns
        table.loc[:, numeric] = table.loc[:, numeric].round(8)
    metrics.to_csv(config.analysis_dir / "implementation_metrics.csv", index=False)
    comparison.to_csv(config.analysis_dir / "pairwise_comparison.csv", index=False)
    predictions.to_csv(config.analysis_dir / "test_predictions.csv", index=False)
    parameter_table.to_csv(
        config.analysis_dir / "logistic_parameter_comparison.csv", index=False
    )
    configuration_table.to_csv(
        config.analysis_dir / "model_configuration.csv", index=False
    )
    regularization_table.to_csv(
        config.analysis_dir / "regularization_path.csv", index=False
    )
    (config.analysis_dir / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )


def run(config: ExperimentConfig) -> Path:
    """Execute the full benchmark and return the compiled PDF path."""

    config.validate()
    for directory in [config.analysis_dir, config.figure_dir, config.pdf_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    X, y, metadata = load_wdbc()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_fraction,
        stratify=y,
        random_state=config.random_state,
    )
    print(
        f"Benchmark: {len(X)} rows, {X.shape[1]} features; "
        f"split={len(X_train)}/{len(X_test)}; positive={metadata.positive_class}"
    )
    pairs = build_pairs(len(X_train), config)
    metrics, comparison, predictions, fitted = fit_pairs(
        pairs, X_train, y_train, X_test, y_test, config
    )
    parameter_table = logistic_parameter_checks(fitted, X_train, y_train)
    configuration_table = model_configuration_table(pairs)
    regularization_table = regularization_diagnostic(
        X_train, y_train, config.random_state
    )

    loss_curves = {
        family.replace("Logistic ", ""): fitted[f"{family}|scratch"]
        .named_steps["model"]
        .loss_curve_
        for family in ["Logistic none", "Logistic L1", "Logistic L2"]
    }
    plot_head_to_head_auc(comparison, config.figure_dir)
    plot_auc_gaps(comparison, config.figure_dir)
    plot_probability_agreement(
        y_test.to_numpy(), predictions, comparison, config.figure_dir
    )
    plot_logistic_optimization(loss_curves, config.figure_dir)
    plot_regularization_effect(regularization_table, config.figure_dir)

    run_metadata = {
        "topic": "tabular",
        "dataset": metadata.name,
        "dataset_source": metadata.source_url,
        "dataset_doi": metadata.doi,
        "dataset_license": metadata.license,
        "task": metadata.task,
        "positive_class": metadata.positive_class,
        "n_rows": len(X),
        "n_features": X.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_fraction": config.test_fraction,
        "split": "fixed class-stratified train/test split",
        "preprocessing": (
            "StandardScaler fitted on training data for logistic models only; "
            "tree models use original numeric features"
        ),
        "random_state": config.random_state,
        "bootstrap_samples": config.n_bootstrap,
        "quick": config.quick,
    }
    write_outputs(
        config,
        metrics,
        comparison,
        predictions,
        parameter_table,
        configuration_table,
        regularization_table,
        run_metadata,
    )

    tex_path = config.pdf_dir / "tabular_models_report.tex"
    write_latex_report(
        comparison,
        metrics,
        parameter_table,
        configuration_table,
        tex_path,
        n_train=len(X_train),
        n_test=len(X_test),
        n_features=X.shape[1],
        n_bootstrap=config.n_bootstrap,
        random_state=config.random_state,
    )
    pdf_path = compile_latex_report(tex_path)
    print("\nPAIR SUMMARY")
    print(
        comparison[
            [
                "family",
                "scratch_roc_auc",
                "library_roc_auc",
                "auc_gap",
                "label_agreement",
                "probability_mae",
                "consistency",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nAnalysis: {config.analysis_dir.resolve()}")
    print(f"Figures: {config.figure_dir.resolve()}")
    print(f"PDF report: {pdf_path.resolve()}")
    return pdf_path


def parse_config() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--random-state", type=int, default=550)
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    config = ExperimentConfig(
        output_root=arguments.output_root,
        n_bootstrap=arguments.bootstrap,
        random_state=arguments.random_state,
        quick=arguments.quick,
    )
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    return config


def main() -> None:
    run(parse_config())


if __name__ == "__main__":
    main()
