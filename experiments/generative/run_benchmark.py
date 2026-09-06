"""Run PyTorch flow-matching and diffusion architecture ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture

from math550.topics.generative import (
    ConditionalFlowMatcher,
    DenoisingDiffusion,
    GaussianVAE,
    evaluate_generator,
    linear_beta_schedule,
    load_latent_digits,
    make_diffusion_training_data,
    make_flow_matching_training_data,
    SUPPORTED_TORCH_DEVICES,
)
from math550.topics.generative.plotting import (
    plot_latent_samples,
    plot_learning_curves,
    plot_metric_comparison,
    plot_sample_grids,
    plot_sampling_tradeoff,
    plot_transport_trajectories,
)
from math550.topics.generative.reporting import compile_latex_report, write_latex_report

from .config import ExperimentConfig, build_regressor_ablation
from .diagnostics import affine_flow_exactness, paired_bootstrap_summary


def _fit_with_diagnostics(
    model: object,
    features: np.ndarray,
    targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
) -> float:
    start = time.perf_counter()
    model.fit(features, targets, validation_features, validation_targets)
    return time.perf_counter() - start


def _build_models(
    config: ExperimentConfig,
    flow_features: np.ndarray,
    flow_targets: np.ndarray,
    flow_validation_features: np.ndarray,
    flow_validation_targets: np.ndarray,
    diffusion_features: np.ndarray,
    diffusion_targets: np.ndarray,
    diffusion_validation_features: np.ndarray,
    diffusion_validation_targets: np.ndarray,
    betas: np.ndarray,
    vae_train: np.ndarray,
    vae_validation: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, list[float]]]:
    models: dict[str, object] = {}
    objective_rows: list[dict[str, object]] = []
    loss_curves: dict[str, list[float]] = {}
    for family, features, targets, validation_features, validation_targets, offset in [
        (
            "Flow matching",
            flow_features,
            flow_targets,
            flow_validation_features,
            flow_validation_targets,
            0,
        ),
        (
            "Diffusion",
            diffusion_features,
            diffusion_targets,
            diffusion_validation_features,
            diffusion_validation_targets,
            100,
        ),
    ]:
        regressors = build_regressor_ablation(config, seed_offset=offset)
        for implementation, regressor in regressors.items():
            if family == "Flow matching":
                model = ConditionalFlowMatcher(
                    regressor,
                    sigma_min=0.01,
                    random_state=config.random_state,
                    name=f"{family} {implementation}",
                )
            else:
                model = DenoisingDiffusion(
                    regressor,
                    betas=betas,
                    random_state=config.random_state,
                    name=f"{family} {implementation}",
                )
            elapsed = _fit_with_diagnostics(
                model,
                features,
                targets,
                validation_features,
                validation_targets,
            )
            key = f"{family}|{implementation}"
            models[key] = model
            curve = [float(value) for value in getattr(regressor, "loss_curve_", [])]
            loss_curves[key] = curve
            objective_rows.append(
                {
                    "family": family,
                    "implementation": implementation,
                    "fit_seconds": elapsed,
                    "epochs": len(curve),
                    "initial_recorded_loss": curve[0] if curve else np.nan,
                    "final_recorded_loss": curve[-1] if curve else np.nan,
                    "training_regression_mse": model.regression_mse(features, targets),
                    "validation_regression_mse": model.regression_mse(
                        validation_features, validation_targets
                    ),
                    "training_objective": model.regression_mse(features, targets),
                    "validation_objective": model.regression_mse(
                        validation_features, validation_targets
                    ),
                    "parameter_count": int(regressor.parameter_count_),
                    "optimizer": "torch.optim.Adam",
                    "device": str(regressor.device_),
                    "nonfinite_detected": False,
                }
            )
            print(
                f"{key:28s} fit={elapsed:7.2f}s "
                f"train_mse={objective_rows[-1]['training_regression_mse']:.4f} "
                f"valid_mse={objective_rows[-1]['validation_regression_mse']:.4f}"
            )

    start = time.perf_counter()
    vae = GaussianVAE(
        latent_dim=min(6, vae_train.shape[1]),
        hidden_sizes=config.hidden_sizes,
        observation_sigma=0.35,
        beta=1.0,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        max_epochs=config.effective_epochs,
        random_state=config.random_state + 200,
        device=config.device,
    ).fit(vae_train, vae_validation)
    elapsed = time.perf_counter() - start
    training_vae = vae.evaluate(vae_train)
    validation_vae = vae.evaluate(vae_validation)
    models["VAE|mlp"] = vae
    loss_curves["VAE|mlp"] = [float(value) for value in vae.loss_curve_]
    objective_rows.append(
        {
            "family": "VAE",
            "implementation": "mlp",
            "fit_seconds": elapsed,
            "epochs": len(vae.loss_curve_),
            "initial_recorded_loss": vae.loss_curve_[0],
            "final_recorded_loss": vae.loss_curve_[-1],
            "training_regression_mse": np.nan,
            "validation_regression_mse": np.nan,
            "training_objective": training_vae["objective"],
            "validation_objective": validation_vae["objective"],
            "training_reconstruction_mse": training_vae["reconstruction_mse"],
            "validation_reconstruction_mse": validation_vae["reconstruction_mse"],
            "mean_validation_kl": validation_vae["kl"],
            "parameter_count": int(vae.parameter_count_),
            "optimizer": "torch.optim.Adam",
            "device": str(vae.device_),
            "nonfinite_detected": False,
        }
    )
    print(
        f"{'VAE|mlp':28s} fit={elapsed:7.2f}s "
        f"train_objective={training_vae['objective']:.4f} "
        f"valid_objective={validation_vae['objective']:.4f}"
    )
    return models, pd.DataFrame(objective_rows), loss_curves


def _sample_model(model: object, family: str, n: int, config: ExperimentConfig, seed: int) -> np.ndarray:
    if family == "Flow matching":
        return model.sample(n, n_steps=config.ode_steps, random_state=seed, solver="rk4")
    if family == "Diffusion":
        return model.sample(
            n, n_sampling_steps=config.diffusion_steps, random_state=seed
        )
    return model.sample(n, random_state=seed, sample_observation=True)


def _evaluate_models(
    models: dict[str, object], data: object, config: ExperimentConfig
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    display_samples: dict[str, np.ndarray] = {}
    timing_rows: list[dict[str, object]] = []
    for key, model in models.items():
        family, implementation = key.split("|")
        for replicate in range(config.effective_replicates):
            seed = config.random_state + 1_000 + replicate
            start = time.perf_counter()
            generated = _sample_model(
                model, family, config.effective_generated, config, seed
            )
            elapsed = time.perf_counter() - start
            metrics = evaluate_generator(
                data.test_latent, data.train_latent, data.train_labels, generated
            )
            rows.append(
                {
                    "family": family,
                    "implementation": implementation,
                    "replicate": replicate,
                    "seed": seed,
                    "sample_seconds": elapsed,
                    **metrics,
                }
            )
            if replicate == 0:
                display_samples[key] = generated[:64]
            timing_rows.append(
                {
                    "family": family,
                    "implementation": implementation,
                    "replicate": replicate,
                    "n_samples": config.effective_generated,
                    "seconds": elapsed,
                    "samples_per_second": config.effective_generated / elapsed,
                }
            )

    mixture = GaussianMixture(
        n_components=10,
        covariance_type="full",
        reg_covar=1e-5,
        max_iter=500,
        random_state=config.random_state,
    ).fit(data.train_latent)
    for replicate in range(config.effective_replicates):
        seed = config.random_state + 1_000 + replicate
        mixture.random_state = seed
        start = time.perf_counter()
        generated = mixture.sample(config.effective_generated)[0]
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "family": "Gaussian mixture",
                "implementation": "classical baseline",
                "replicate": replicate,
                "seed": seed,
                "sample_seconds": elapsed,
                **evaluate_generator(
                    data.test_latent, data.train_latent, data.train_labels, generated
                ),
            }
        )
        if replicate == 0:
            display_samples["Gaussian mixture|classical baseline"] = generated[:64]
    return pd.DataFrame(rows), display_samples, pd.DataFrame(timing_rows)


def _sampling_tradeoff(
    models: dict[str, object], data: object, config: ExperimentConfig
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sample_count = 240 if config.quick else 500
    for family, steps in [
        ("Flow matching", [5, 10, 20, config.ode_steps]),
        ("Diffusion", [5, 10, 20, config.diffusion_steps]),
    ]:
        model = models[f"{family}|mlp"]
        for n_steps in sorted(set(steps)):
            start = time.perf_counter()
            if family == "Flow matching":
                generated = model.sample(
                    sample_count,
                    n_steps=n_steps,
                    random_state=config.random_state + 2_000,
                    solver="rk4",
                )
                network_evaluations = 4 * n_steps
            else:
                generated = model.sample(
                    sample_count,
                    n_sampling_steps=n_steps,
                    random_state=config.random_state + 2_000,
                )
                network_evaluations = n_steps
            elapsed = time.perf_counter() - start
            metrics = evaluate_generator(
                data.test_latent, data.train_latent, data.train_labels, generated
            )
            rows.append(
                {
                    "family": family,
                    "steps": n_steps,
                    "network_evaluations": network_evaluations,
                    "seconds": elapsed,
                    "frechet_latent": metrics["frechet_latent"],
                    "mmd_rbf": metrics["mmd_rbf"],
                }
            )
    vae = models["VAE|mlp"]
    start = time.perf_counter()
    generated = vae.sample(
        sample_count,
        random_state=config.random_state + 2_000,
        sample_observation=True,
    )
    elapsed = time.perf_counter() - start
    metrics = evaluate_generator(
        data.test_latent, data.train_latent, data.train_labels, generated
    )
    rows.append(
        {
            "family": "VAE",
            "steps": 1,
            "network_evaluations": 1,
            "seconds": elapsed,
            "frechet_latent": metrics["frechet_latent"],
            "mmd_rbf": metrics["mmd_rbf"],
        }
    )
    return pd.DataFrame(rows)


def run(config: ExperimentConfig) -> Path:
    config.validate()
    for directory in [config.analysis_dir, config.figure_dir, config.pdf_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    data = load_latent_digits(
        n_components=config.n_components,
        test_fraction=config.test_fraction,
        random_state=config.random_state,
    )
    n_train_pairs = config.effective_training_pairs
    n_validation_pairs = config.effective_validation_pairs
    flow_train = make_flow_matching_training_data(
        data.train_latent, n_train_pairs, config.random_state, sigma_min=0.01
    )
    flow_validation = make_flow_matching_training_data(
        data.test_latent, n_validation_pairs, config.random_state + 1, sigma_min=0.01
    )
    betas = linear_beta_schedule(config.diffusion_steps)
    diffusion_train = make_diffusion_training_data(
        data.train_latent, n_train_pairs, betas, config.random_state + 2
    )
    diffusion_validation = make_diffusion_training_data(
        data.test_latent, n_validation_pairs, betas, config.random_state + 3
    )

    models, objectives, loss_curves = _build_models(
        config,
        *flow_train,
        *flow_validation,
        *diffusion_train,
        *diffusion_validation,
        betas,
        data.train_latent,
        data.test_latent,
    )

    metrics, display_samples, timing = _evaluate_models(models, data, config)
    comparison = paired_bootstrap_summary(
        metrics, config.bootstrap_samples, config.random_state
    )
    exactness = affine_flow_exactness(data.train_latent, data.test_latent)
    tradeoff = _sampling_tradeoff(models, data, config)

    objectives.to_csv(config.analysis_dir / "training_objectives.csv", index=False)
    metrics.to_csv(config.analysis_dir / "sample_metrics.csv", index=False)
    comparison.to_csv(config.analysis_dir / "paired_comparison.csv", index=False)
    exactness.to_csv(config.analysis_dir / "affine_flow_exactness.csv", index=False)
    tradeoff.to_csv(config.analysis_dir / "sampling_tradeoff.csv", index=False)
    timing.to_csv(config.analysis_dir / "sampling_runtime.csv", index=False)

    run_metadata = {
        "topic": "generative",
        "dataset": data.metadata.name,
        "dataset_source": data.metadata.source_url,
        "dataset_doi": data.metadata.doi,
        "dataset_license": data.metadata.license,
        "n_rows": data.metadata.n_rows,
        "n_train": len(data.train_latent),
        "n_test": len(data.test_latent),
        "image_shape": data.metadata.image_shape,
        "latent_components": config.n_components,
        "pca_explained_variance": float(data.pca.explained_variance_ratio_.sum()),
        "pca_fit": "training rows only; whitening enabled",
        "training_pairs": n_train_pairs,
        "validation_pairs": n_validation_pairs,
        "epochs": config.effective_epochs,
        "hidden_sizes": config.hidden_sizes,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "flow_path": "independent Gaussian-data coupling; sigma_min=0.01",
        "flow_sampler": f"RK4 with {config.ode_steps} steps",
        "diffusion_schedule": (
            f"linear beta 1e-4 to 0.02 over {config.diffusion_steps} steps"
        ),
        "diffusion_sampler": "deterministic DDIM (eta=0)",
        "vae_objective": (
            "Gaussian decoder negative ELBO up to an additive constant; "
            "six-dimensional diagonal-Gaussian posterior; beta=1"
        ),
        "vae_sampler": "one prior draw and one decoder evaluation",
        "evaluation_replicates": config.effective_replicates,
        "generated_per_replicate": config.effective_generated,
        "bootstrap_samples": config.bootstrap_samples,
        "random_state": config.random_state,
        "quick": config.quick,
        "framework": "PyTorch",
        "torch_version": torch.__version__,
        "requested_device": config.device,
        "device": sorted(objectives["device"].unique().tolist()),
        "architecture_comparison": (
            "flow and diffusion compare a two-hidden-layer tanh MLP with a single "
            "linear layer under matched regression settings; VAE uses a two-hidden-layer "
            "encoder/decoder and the same epoch, batch, learning-rate, and device policy"
        ),
        "metric_scope": (
            "All distributional metrics are computed in the declared training-fitted "
            "whitened PCA space; frechet_latent is not image-feature FID."
        ),
    }
    visual_examples_path = config.analysis_dir / "visual_examples.json"
    visual_examples = (
        json.loads(visual_examples_path.read_text(encoding="utf-8"))
        if visual_examples_path.exists()
        else {}
    )
    run_metadata["visual_examples_evidence"] = (
        str(visual_examples_path) if visual_examples else "not run"
    )
    (config.analysis_dir / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )

    plot_learning_curves(loss_curves, config.figure_dir)
    plot_metric_comparison(metrics, config.figure_dir)
    plot_sampling_tradeoff(tradeoff, config.figure_dir)
    plot_latent_samples(data.test_latent, display_samples, config.figure_dir)
    plot_sample_grids(data, display_samples, config.figure_dir)
    _, flow_path = models["Flow matching|mlp"].sample(
        36,
        n_steps=config.ode_steps,
        random_state=config.random_state + 2_500,
        solver="rk4",
        return_path=True,
    )
    plot_transport_trajectories(data.train_latent, flow_path, config.figure_dir)

    tex_path = config.pdf_dir / "generative_models_report.tex"
    write_latex_report(
        objectives=objectives,
        metrics=metrics,
        comparison=comparison,
        exactness=exactness,
        tradeoff=tradeoff,
        run_metadata=run_metadata,
        output_path=tex_path,
        visual_examples=visual_examples,
    )
    pdf_path = compile_latex_report(tex_path)
    print("\nMEAN SAMPLE METRICS")
    print(
        metrics.groupby(["family", "implementation"])[
            ["frechet_latent", "mmd_rbf", "precision_knn", "recall_knn", "label_entropy"]
        ]
        .mean()
        .round(4)
        .to_string()
    )
    print(f"\nReport: {pdf_path.resolve()}")
    return pdf_path


def parse_config() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument(
        "--device",
        choices=SUPPORTED_TORCH_DEVICES,
        default="auto",
        help="PyTorch device policy (auto prefers CUDA, then MPS, then CPU)",
    )
    arguments = parser.parse_args()
    return ExperimentConfig(
        output_root=arguments.output_root,
        quick=arguments.quick,
        max_epochs=arguments.epochs,
        bootstrap_samples=arguments.bootstrap,
        device=arguments.device,
    )


if __name__ == "__main__":
    run(parse_config())
