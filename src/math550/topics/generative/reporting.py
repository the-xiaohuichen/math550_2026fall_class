"""Generate and compile the evidence-backed generative-model course report."""

from __future__ import annotations

from datetime import date
import os
from pathlib import Path
import subprocess

import pandas as pd


def _escape(value: object) -> str:
    text = str(value)
    for source, target in {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }.items():
        text = text.replace(source, target)
    return text


def _mean_metric_rows(metrics: pd.DataFrame) -> str:
    summary = (
        metrics.groupby(["family", "implementation"])[
            [
                "frechet_latent",
                "mmd_rbf",
                "precision_knn",
                "recall_knn",
                "label_entropy",
                "class_coverage",
                "median_nearest_train",
            ]
        ]
        .mean()
        .reset_index()
    )
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            f"{_escape(row['family'])} & {_escape(row['implementation'])} & "
            f"{row['frechet_latent']:.3f} & {row['mmd_rbf']:.4f} & "
            f"{row['precision_knn']:.3f} & {row['recall_knn']:.3f} & "
            f"{row['label_entropy']:.3f} & {row['class_coverage']:.1f} & "
            f"{row['median_nearest_train']:.3f} \\\\"
        )
    return "\n".join(rows)


def _objective_rows(objectives: pd.DataFrame) -> str:
    rows = []
    for _, row in objectives.iterrows():
        rows.append(
            f"{_escape(row['family'])} & {_escape(row['implementation'])} & "
            f"{int(row['epochs'])} & {int(row['parameter_count'])} & "
            f"{row['training_objective']:.4f} & "
            f"{row['validation_objective']:.4f} & {row['fit_seconds']:.2f} \\\\"
        )
    return "\n".join(rows)


def _comparison_rows(comparison: pd.DataFrame) -> str:
    selected = comparison[
        comparison["metric"].isin(
            ["frechet_latent", "mmd_rbf", "precision_knn", "recall_knn"]
        )
    ]
    rows = []
    for _, row in selected.iterrows():
        rows.append(
            f"{_escape(row['family'])} & {_escape(row['metric'])} & "
            f"{row['mlp_mean']:.4f} & {row['linear_mean']:.4f} & "
            f"{row['gap']:+.4f} [{row['gap_ci_low']:+.4f}, "
            f"{row['gap_ci_high']:+.4f}] \\\\"
        )
    return "\n".join(rows)


def _tradeoff_rows(tradeoff: pd.DataFrame) -> str:
    rows = []
    for _, row in tradeoff.sort_values(["family", "network_evaluations"]).iterrows():
        rows.append(
            f"{_escape(row['family'])} & {int(row['steps'])} & "
            f"{int(row['network_evaluations'])} & {row['seconds']:.3f} & "
            f"{row['frechet_latent']:.3f} & {row['mmd_rbf']:.4f} \\\\"
        )
    return "\n".join(rows)


def write_latex_report(
    objectives: pd.DataFrame,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    exactness: pd.DataFrame,
    tradeoff: pd.DataFrame,
    run_metadata: dict[str, object],
    output_path: Path,
    visual_examples: dict[str, object] | None = None,
) -> None:
    """Write a graduate-level report whose numerical claims come from saved outputs."""

    summary = (
        metrics.groupby(["family", "implementation"])[
            ["frechet_latent", "mmd_rbf", "precision_knn", "recall_knn", "label_entropy"]
        ]
        .mean()
        .reset_index()
    )
    best_frechet = summary.loc[summary["frechet_latent"].idxmin()]
    exact = exactness.iloc[0]
    flow_gap = comparison[
        (comparison["family"] == "Flow matching")
        & (comparison["metric"] == "frechet_latent")
    ].iloc[0]
    diffusion_gap = comparison[
        (comparison["family"] == "Diffusion")
        & (comparison["metric"] == "frechet_latent")
    ].iloc[0]
    requested_device = _escape(run_metadata.get("requested_device", "not recorded"))
    device_value = run_metadata.get("device", "not recorded")
    if isinstance(device_value, (list, tuple)):
        resolved_device = _escape(", ".join(str(value) for value in device_value))
    else:
        resolved_device = _escape(device_value)
    pca_variance = 100.0 * float(run_metadata["pca_explained_variance"])
    visual_examples = visual_examples or {}
    moons = visual_examples.get("two_moons")
    moons_vae = visual_examples.get("two_moons_vae")
    mnist = visual_examples.get("mnist_28")
    mnist_vae = visual_examples.get("mnist_28_vae")
    if isinstance(moons, dict):
        moons_block = rf"""
\subsection*{{4.3 Code 1 made visible: Gaussian noise to two moons}}

The first visual demonstration follows Code 1 on page 7 of the supplied guide
\cite{{lipman2024guide}}: \code{{make\_moons(256, noise=0.05)}}, fresh standard
Gaussian source particles, uniform times, the straight interpolation
$x_t=(1-t)x_0+tx_1$, and target velocity $x_1-x_0$. The PyTorch vector field has
three width-64 ELU hidden layers and {_escape(moons['parameter_count'])} parameters.
Adam uses learning rate $10^{{-2}}$ for {_escape(moons['updates'])} updates. Generation
uses only eight midpoint ODE steps.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.91\linewidth]{{two_moons_flow_path.png}}
\caption{{A common set of particles evolves from an isotropic Gaussian into the
two-moons geometry. Gray points are the target reference at every time, so shape
formation is directly visible rather than inferred from a scalar metric.}}
\end{{figure}}

On fixed evaluation draws, RBF MMD$^2$ was {moons['rbf_mmd_squared']:.5f};
{100.0 * moons['generated_support_fraction']:.1f}\% of generated points fell within
the support threshold defined by the 95th percentile held-out-to-reference nearest
distance. The generated median nearest-reference distance was
{moons['median_nearest_reference']:.4f}, versus
{moons['heldout_median_nearest_reference']:.4f} for an independent held-out draw.
These values and the plot establish a fixed-seed pipeline/geometric demonstration,
not general superiority of flow matching.
"""
    else:
        moons_block = ""
    if isinstance(moons_vae, dict):
        moons_vae_block = rf"""
\subsection*{{4.4 The same two moons expose the VAE tradeoff}}

The VAE uses a two-dimensional diagonal-Gaussian posterior, a standard-normal
prior, and an isotropic Gaussian decoder with fixed standard deviation $0.05$.
Both encoder and decoder are two-layer width-64 ELU PyTorch MLPs. Its minimized
objective is the negative ELBO up to the fixed Gaussian normalizing constant:
the scaled reconstruction cost plus the analytic posterior-to-prior KL.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\linewidth]{{two_moons_vae.png}}
\caption{{Observed points, posterior means, prior draws, and one-pass decoded
samples. The aggregate encoded geometry does not perfectly match the isotropic
prior, so prior sampling fills the gap between the two thin moons.}}
\end{{figure}}

After {_escape(moons_vae['epochs'])} epochs, the held-out reconstruction MSE was
{moons_vae['heldout_reconstruction_mse']:.6f} and mean KL was
{moons_vae['heldout_mean_kl']:.3f} nats. Yet sample MMD$^2$ was
{moons_vae['rbf_mmd_squared']:.5f}, and only
{100.0 * moons_vae['generated_support_fraction']:.1f}\% of generated points fell
inside the held-out support threshold. This distinction is pedagogically useful:
good reconstructions do not imply that samples from $p(z)$ land on the target
manifold. The VAE generates with one decoder evaluation, while flow matching
uses repeated vector-field evaluations.
"""
    else:
        moons_vae_block = ""
    if isinstance(mnist, dict):
        mnist_block = rf"""
\subsection*{{7.1 Original MNIST at native 28-by-28 resolution}}

The second visual demonstration uses the original MNIST split: 60,000 training
and 10,000 test images at 28-by-28 resolution \cite{{mnistOriginal}}. All four
compressed IDX files are MD5-verified before parsing. The official distribution
page does not state an explicit dataset license, which is recorded rather than
silently replaced by a mirror's terms. The model uses all
{_escape(mnist['n_training_rows_used'])} training rows.

The generator is class-conditional so students can see that a declared condition
changes the output distribution. A compact {_escape(mnist['parameter_count'])}-parameter
U-Net-style convolutional vector field is composed entirely from standard PyTorch
layers; the course code implements the flow target, optimization loop, exponential
moving average, and midpoint sampler explicitly. It trained for
{_escape(mnist['epochs'])} epochs and sampled with {_escape(mnist['sampling_steps'])}
midpoint steps. The requested device was \code{{{_escape(mnist['requested_device'])}}}
and the resolved device was \code{{{_escape(mnist['resolved_device'])}}}.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.93\linewidth]{{mnist_28_generated.png}}
\caption{{Eighty generated 28-by-28 images. Columns 0--9 are requested class labels;
rows begin from independent Gaussian draws. This exposes both controllability and
within-label variation, including visible failures.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.86\linewidth]{{mnist_28_flow_progression.png}}
\caption{{Selected particles along one 20-step midpoint trajectory. Global digit
structure and background suppression emerge continuously from noise.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.78\linewidth]{{mnist_28_training_curve.png}}
\caption{{Online flow-matching MSE per pixel. The held-out curve uses fixed noise/time
pairs over 2,000 official test images and is an objective diagnostic, not a sample-
quality metric.}}
\end{{figure}}

Final training and held-out velocity MSE were
{mnist['final_training_velocity_mse']:.3f} and
{mnist['final_test_pair_velocity_mse']:.3f}. A 3-neighbor probe trained on 20,000
downsampled training images scored {100.0 * mnist['knn_probe_test_accuracy']:.1f}\%
on 2,000 original test images, then agreed with requested labels for
{100.0 * mnist['requested_label_accuracy_under_probe']:.1f}\% of generated samples.
The probe makes conditional failures countable but is not human judgment or FID.
Generated median nearest-training distance was
{mnist['generated_median_nearest_train_14x14']:.3f}, compared with
{mnist['heldout_median_nearest_train_14x14']:.3f} for held-out images in the same
14-by-14 feature space; this does not indicate exact copying, but it is not a
privacy audit.
"""
    else:
        mnist_block = ""
    if isinstance(mnist_vae, dict):
        mnist_vae_block = rf"""
\subsection*{{7.2 A conditional convolutional VAE on the same MNIST split}}

The second MNIST model uses the same official split and label conditions but a
different probabilistic contract. The encoder returns a 16-dimensional diagonal
Gaussian $q_\phi(z\mid x,y)$; the decoder models independent Bernoulli pixels
$p_\theta(x\mid z,y)$. The {_escape(mnist_vae['parameter_count'])}-parameter
network is built from standard PyTorch convolution, transposed-convolution,
normalization, activation, and linear layers. Generation draws $z\sim\mathcal
N(0,I)$ and uses one decoder pass.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.93\linewidth]{{mnist_28_vae_generated.png}}
\caption{{Eighty conditional VAE samples from prior draws. Columns identify the
requested digit. Samples are recognizable but visibly smoother than many flow
outputs, reflecting the Bernoulli reconstruction geometry and latent bottleneck.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.94\linewidth]{{mnist_28_vae_reconstruction.png}}
\caption{{Held-out images and posterior-mean reconstructions. This diagnostic
tests the encoder-decoder path; it does not test prior sampling.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\linewidth]{{mnist_28_vae_training_curve.png}}
\caption{{Negative ELBO and its reconstruction/KL components. Reporting both
terms prevents a decreasing total from hiding posterior collapse or a reduction
error.}}
\end{{figure}}

After {_escape(mnist_vae['epochs'])} epochs, held-out negative ELBO was
{mnist_vae['heldout_negative_elbo']:.2f} nats per image: reconstruction BCE
{mnist_vae['heldout_reconstruction_bce']:.2f} plus KL
{mnist_vae['heldout_kl']:.2f}. The same style of 3-neighbor probe scored
{100.0 * mnist_vae['knn_probe_test_accuracy']:.1f}\% on held-out real digits and
agreed with requested labels for
{100.0 * mnist_vae['requested_label_accuracy_under_probe']:.1f}\% of VAE samples.
The VAE generated median nearest-training distance was
{mnist_vae['generated_median_nearest_train_14x14']:.3f}, versus
{mnist_vae['heldout_median_nearest_train_14x14']:.3f} for held-out images in the
same feature space. These are conditional-control and anomaly screens, not human
recognizability, FID, or privacy guarantees.
"""
    else:
        mnist_vae_block = ""

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,graphicx,float,longtable,microtype,tabularx,xcolor}}
\usepackage[colorlinks=true,linkcolor=blue!45!black,urlcolor=blue!45!black,citecolor=blue!45!black]{{hyperref}}
\hypersetup{{
  pdftitle={{Generative Modeling: Variational Autoencoders, Flows, and Diffusion}},
  pdfauthor={{MATH 550 Capstone Course Development}},
  pdfsubject={{Three-week graduate module with reproducible public-data validation}}
}}
\graphicspath{{{{../generative/figures/}}}}
\definecolor{{navy}}{{HTML}}{{1F4E78}}
\definecolor{{lightblue}}{{HTML}}{{D9EAF7}}
\definecolor{{warm}}{{HTML}}{{F5E7D3}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{4pt}}
\newcommand{{\code}}[1]{{\texttt{{#1}}}}
\newcommand{{\E}}{{\mathbb{{E}}}}

\begin{{document}}

\begin{{center}}
{{\LARGE\bfseries\color{{navy}} Generative Modeling}}\\[4pt]
{{\Large Variational Autoencoders, Flows, and Diffusion}}\\[9pt]
{{\large A three-week graduate capstone module with reproducible evidence}}\\[10pt]
Prepared for MATH 550 Mathematical Data Science Capstone\\
Verified build: {date.today().isoformat()}
\end{{center}}

\colorbox{{lightblue}}{{\parbox{{0.96\linewidth}}{{
\textbf{{Executive finding.}} The module now makes three generative contracts
executable: a VAE optimizes a variational bound and decodes in one pass; flow
matching learns continuous transport; diffusion learns to reverse corruption. On the
public digits benchmark, the lowest mean latent-space Fr\'echet distance was
{_escape(best_frechet['family'])} ({_escape(best_frechet['implementation'])},
{best_frechet['frechet_latent']:.3f}). The flow-matching MLP-minus-linear
gap was {flow_gap['gap']:+.3f}; the diffusion capacity gap was
{diffusion_gap['gap']:+.3f}.
These are compact CPU demonstrations, not image-generation leaderboard claims.
The strongest caveat is the eight-dimensional PCA bottleneck, which retains
{pca_variance:.1f}\% of training-set variance and visibly limits sample sharpness.
Two-moons and original 28-by-28 MNIST demonstrations compare VAE and flow behavior
on the same data, while the lower-resolution benchmark supplies paired metrics.
}}}}

\section*{{1. Communication job and learning outcomes}}

This module is designed for graduate data-science, AI, and machine-learning
capstone students who already know multivariable calculus, probability,
linear algebra, and basic neural-network training. It is not an LLM unit.
After nine 60-minute units, students should be able to:

\begin{{enumerate}}
\item distinguish explicit-likelihood, latent-variable, adversarial, score-based,
and transport-based generators by objective, inference, and sampling cost;
\item derive the change-of-variables, ELBO, adversarial, denoising, and conditional
flow-matching objectives and state what each objective does \emph{{not}} guarantee;
\item construct a conditional probability path and its sample-wise velocity target;
\item define PyTorch encoder/decoder and time-conditioned modules, train them with
autograd and \code{{torch.optim}}, and run one-pass, ODE, or deterministic diffusion samplers;
\item diagnose fidelity, coverage, memorization risk, numerical error, and metric
misuse using multiple forms of evidence; and
\item justify a generative-model family for a capstone application under data,
compute, evaluation, uncertainty, and governance constraints.
\end{{enumerate}}

\section*{{2. Three-week teaching sequence (nine units)}}

\begin{{longtable}}{{@{{}}p{{0.08\linewidth}}p{{0.25\linewidth}}p{{0.42\linewidth}}p{{0.179\linewidth}}@{{}}}}
\toprule
Unit & Central question & Concepts and in-class work & Evidence of learning \\
\midrule
\endhead
1 & What does it mean to learn a distribution? & Pushforwards, density models,
implicit models, likelihood, sampling, conditioning; fidelity versus coverage;
why visual inspection is insufficient. & Metric critique and failure-mode sort. \\
2 & What is gained and lost by latent variables? & Autoregressive factorization,
VAE ELBO and reparameterization, amortized inference, posterior collapse; short
comparison with energy-based models. & ELBO derivation and model-selection memo. \\
3 & Why can adversarial training look good but fail silently? & GAN minimax game,
non-saturating loss, discriminator as density-ratio signal, instability, mode
collapse, evaluation and governance. & Two-distribution game exercise. \\
\midrule
4 & How can an invertible map model density exactly? & Change of variables,
Jacobian determinants, coupling layers, Real NVP, exact inverse and likelihood. &
Affine-coupling round-trip test. \\
5 & What changes in continuous time? & Neural ODEs, continuous normalizing flows,
continuity equation, instantaneous change of variables, Euler and RK4 error. &
ODE solver convergence lab. \\
6 & How does flow matching remove simulation from training? & Probability paths,
conditional paths, marginalization, independent coupling, OT coupling, velocity
regression, guidance and conditioning. & Implement conditional FM targets. \\
\midrule
7 & How does diffusion learn to reverse noise? & Forward Gaussian chain, schedules,
$\epsilon$/$x_0$/velocity parameterizations, denoising loss, ancestral sampling,
DDIM. & Derive forward marginal and code noise targets. \\
8 & Are diffusion and flow matching different languages for the same object? &
Scores, reverse-time SDE, probability-flow ODE, stochastic versus deterministic
samplers, path geometry, network evaluations. & Map formulas and compare samplers. \\
9 & What evidence justifies deployment? & Paired benchmark, sample quality and
coverage, memorization, conditional evaluation, reproducibility, data rights,
failure reporting, capstone design review. & Reproducible audit and decision memo. \\
\bottomrule
\end{{longtable}}

\textbf{{Suggested weekly rhythm.}} Each week uses roughly 100 minutes of
derivation, 55 minutes of live code/diagnostics, and 25 minutes of critique or
design work. The slide deck supplies three visible weekly sections; the report is
the technical record; the homework transfers the method to Fashion-MNIST.

\section*{{3. A compact map of generative-model families}}

\begin{{table}}[H]
\centering
\small
\begin{{tabularx}}{{\linewidth}}{{p{{0.14\linewidth}}p{{0.22\linewidth}}p{{0.18\linewidth}}p{{0.18\linewidth}}X}}
\toprule
Family & Training signal & Density & Sampling & Main teaching value / failure mode \\
\midrule
Autoregressive & exact next-element log likelihood & exact & sequential & exposes
factorization choices; slow sampling and order dependence \\
VAE & ELBO with amortized posterior & lower bound & one decoder pass & clean latent
inference; variational gap and posterior collapse \\
GAN & discriminator--generator game & implicit & one generator pass & direct sample
quality; unstable game and mode dropping \\
Normalizing flow & exact change-of-variables likelihood & exact & invertible pass &
auditable density/inverse; architectural Jacobian constraints \\
Diffusion / score & denoising or score regression & variational / ODE options &
iterative SDE/ODE & stable corruption task; many network evaluations \\
Flow matching & conditional velocity regression & CNF likelihood available & ODE &
simulation-free training and flexible paths; solver/path error still matters \\
Energy-based & unnormalized energy / score & partition function difficult & MCMC &
general compatibility modeling; expensive mixing and normalization \\
\bottomrule
\end{{tabularx}}
\caption{{No family dominates every axis. ``Popular'' is not a sufficient selection criterion.}}
\end{{table}}

\subsection*{{3.1 Objectives students should be able to derive}}

For an invertible map $x=f_\theta(z)$ with base density $p_Z$,
\[
 \log p_X(x)=\log p_Z(f_\theta^{{-1}}(x))
 +\log\left|\det J_{{f_\theta^{{-1}}}}(x)\right|.
\]
The VAE evidence lower bound is
\[
 \log p_\theta(x)\geq
 \E_{{q_\phi(z\mid x)}}[\log p_\theta(x\mid z)]
 -D_{{\mathrm{{KL}}}}(q_\phi(z\mid x)\Vert p(z)).
\]
The original GAN game is
\[
 \min_G\max_D\;\E_{{x\sim p_{{\rm data}}}}\log D(x)
 +\E_{{z\sim p(z)}}\log(1-D(G(z))).
\]
These objectives optimize different surrogates. Likelihood, perceptual fidelity,
coverage, and usefulness downstream are related but not interchangeable.

\subsection*{{3.2 VAE derivation: bound, Gaussian KL, and pathwise gradients}}

Introduce any normalized encoder $q_\phi(z\mid x)$ into the marginal likelihood:
\begin{{align*}}
\log p_\theta(x)
&=\log\int q_\phi(z\mid x)
\frac{{p_\theta(x,z)}}{{q_\phi(z\mid x)}}\,dz\\
&\ge \E_{{q_\phi(z\mid x)}}
\left[\log p_\theta(x,z)-\log q_\phi(z\mid x)\right]\\
&=\E_q[\log p_\theta(x\mid z)]
-D_{{\mathrm{{KL}}}}(q_\phi(z\mid x)\Vert p(z))
=:\mathcal L_{{\rm ELBO}}(x).
\end{{align*}}
The inequality is Jensen's inequality. An exact identity names the variational gap:
\[
\log p_\theta(x)=\mathcal L_{{\rm ELBO}}(x)
+D_{{\mathrm{{KL}}}}(q_\phi(z\mid x)\Vert p_\theta(z\mid x)).
\]
Thus the bound can be loose even when optimization is numerically correct.

For $q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),
\operatorname{{diag}}\sigma_\phi^2(x))$ and $p(z)=\mathcal N(0,I)$,
\[
D_{{\mathrm{{KL}}}}(q_\phi\Vert p)
=\frac12\sum_j\left(\mu_j^2+e^{{\log\sigma_j^2}}-1-\log\sigma_j^2\right).
\]
Sampling is written as the differentiable path
\[
\epsilon\sim\mathcal N(0,I),\qquad
z=\mu_\phi(x)+\exp\!\left(\tfrac12\log\sigma_\phi^2(x)\right)\odot\epsilon.
\]
Conditioned on $\epsilon$, gradients pass through $z$ to both encoder and decoder.
The classroom code minimizes one Monte Carlo estimate of the negative ELBO. For
continuous two-moons/PCA observations it uses a fixed-variance Gaussian decoder;
for MNIST it uses independent Bernoulli pixels, so reconstruction is binary
cross-entropy. Reconstructions diagnose $q\to p_\theta(x\mid z)$; prior samples
diagnose $p(z)\to p_\theta(x\mid z)$. Neither alone validates the other.

\section*{{4. From normalizing flows to flow matching}}

\subsection*{{4.1 Continuous transport}}

A time-dependent velocity $u_t$ defines a flow through
\[
 \frac{{d}}{{dt}}X_t=u_t(X_t),\qquad X_0\sim p_0.
\]
The associated density obeys the continuity equation
$\partial_t p_t+\nabla\!\cdot(p_tu_t)=0$. Continuous normalizing flows can track
density along trajectories through
\[
 \frac{{d}}{{dt}}\log p_t(X_t)=-\nabla\!\cdot u_t(X_t),
\]
while sample generation needs only the state ODE. Neural ODE solvers expose a
real modeling tradeoff: smaller steps reduce discretization error but require
more network evaluations \cite{{chen2018neural}}.

\subsection*{{4.2 Conditional flow matching}}

Following the supplied guide \cite{{lipman2024guide}} and the original flow-
matching paper \cite{{lipman2023flow}}, the global regression objective is
\[
 \mathcal L_{{\rm FM}}(\theta)=
 \E_{{t,X_t}}\lVert v_\theta(t,X_t)-u_t(X_t)\rVert_2^2.
\]
The marginal field is generally unavailable. Conditional flow matching samples a
data endpoint $X_1$ and a tractable conditional path; its expected gradient equals
the marginal objective gradient under the stated construction. The classroom code
uses independent coupling, $X_0\sim\mathcal N(0,I)$ and $X_1$ sampled from data:
\[
 X_t=[1-(1-\sigma_{{\min}})t]X_0+tX_1,
 \qquad u_t=X_1-(1-\sigma_{{\min}})X_0.
\]
Training is ordinary vector regression; generation solves the learned ODE from a
new noise draw. The simplicity is the lesson, not evidence that path selection is
irrelevant. Independent couplings can induce crossing or curved marginal flows;
minibatch optimal-transport couplings may straighten paths but add estimation and
batch dependence.

{moons_block}
{moons_vae_block}

\section*{{5. Diffusion and score-based modeling}}

For a variance-preserving discrete diffusion,
\[
 q(x_t\mid x_0)=\mathcal N(\sqrt{{\bar\alpha_t}}x_0,
 (1-\bar\alpha_t)I),\qquad
 x_t=\sqrt{{\bar\alpha_t}}x_0+\sqrt{{1-\bar\alpha_t}}\epsilon.
\]
The classroom model predicts the injected noise using
\[
 \mathcal L_\epsilon(\theta)=
 \E_{{t,x_0,\epsilon}}\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2.
\]
This is connected to the weighted variational objective and denoising score
matching in DDPM \cite{{ho2020ddpm}}. In continuous time, a forward SDE that adds
noise has a reverse-time SDE whose drift depends on the score
$\nabla_x\log p_t(x)$; an associated probability-flow ODE shares the same
marginals \cite{{song2021sde}}. Thus diffusion and flow methods can share network
parameterizations and deterministic ODE samplers while differing in path,
regression target, weighting, and stochasticity.

The benchmark uses a linear 50-step beta schedule and deterministic DDIM-style
updates ($\eta=0$). This makes network-evaluation cost explicit and keeps sampling
reproducible. It omits U-Nets, attention, latent autoencoders, classifier-free
guidance, and high-resolution training; those are extensions after the mathematical
core is auditable.

\section*{{6. Reusable implementation and controlled comparison}}

The repository separates reusable models from orchestration. Neural-network
primitives are not reimplemented. Flow and diffusion use time-conditioned
\code{{nn.Module}} regressors with standard \code{{Linear}} and activation layers.
The VAE uses standard PyTorch encoders, posterior heads, decoders, convolutional
layers, binary cross-entropy, and the analytic Gaussian KL. Every wrapper keeps
tensor shapes, minibatch loops, train/evaluation modes, validation, autograd, and
optimizer updates explicit \cite{{pytorchDocs}}.

The run requested device \code{{{requested_device}}}; every fitted architecture
reported resolved device \code{{{resolved_device}}}. The command-line policy also
accepts \code{{auto}}, \code{{cuda}}, and \code{{mps}}. \code{{auto}} prefers
CUDA, then Apple Metal, then CPU; explicit unavailable accelerators raise an
error rather than silently changing hardware.

The controlled flow/diffusion comparison is an architecture ablation, not a
duplicate neural implementation. A two-hidden-layer tanh MLP is compared with a
single linear map under identical target tables, seeds, MSE loss, optimizer,
batch size, learning rate, and epoch budget. The VAE is an additional model-family
comparison with a different objective and sampler; its row must not be read as a
matched capacity intervention.

\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{llrrrrr}}
\toprule
Family & Architecture & Epochs & Params & Train objective & Valid. objective & Fit sec. \\
\midrule
{_objective_rows(objectives)}
\bottomrule
\end{{tabular}}
\caption{{PyTorch objective diagnostics. Flow/diffusion rows are per-coordinate
MSE; VAE values are negative ELBO up to the declared Gaussian constant. Objectives
are meaningful within a family, not directly comparable across rows.}}
\end{{table}}

The normalizing-flow unit has a stricter numerical oracle. The affine flow's
change-of-variables log density agrees with SciPy's multivariate Gaussian to
maximum absolute gap {exact['max_log_density_abs_gap']:.2e}; its maximum
round-trip error is {exact['max_round_trip_abs_gap']:.2e}, and forward/inverse
log determinants cancel to {exact['max_logdet_cancellation_gap']:.2e}. An
additional fixed Real-NVP-style coupling layer is unit-tested for exact inverse
and log-Jacobian cancellation \cite{{dinh2017realnvp}}.

\section*{{7. Public-data protocol}}

The empirical demonstration uses scikit-learn's 1,797-row copy of the UCI Optical
Recognition of Handwritten Digits data: 8-by-8 integer images with values 0--16,
ten digit classes, and no missing values. The UCI source assigns DOI
\href{{https://doi.org/10.24432/C50P49}}{{10.24432/C50P49}} and CC BY 4.0
licensing \cite{{uciDigits}}. A fixed stratified split has
{run_metadata['n_train']} training and {run_metadata['n_test']} held-out rows.

PCA is fitted \emph{{only}} on training images, uses whitening, and retains eight
components. The transformation is a declared capacity/compute choice, not an
innocent display operation. It retains {pca_variance:.1f}\% of training variance,
and all training and evaluation occurs in that declared latent space. Therefore:

\begin{{itemize}}
\item decoded sample blur partly reflects discarded PCA information;
\item ``latent Fr\'echet distance'' compares Gaussian moments in this PCA space
and is not Inception-feature FID;
\item the fixed $k$-NN probe estimates digit diversity but is not a human fidelity
study; and
\item results validate code paths and expose tradeoffs; they do not rank modern
high-resolution architectures.
\end{{itemize}}

{mnist_block}
{mnist_vae_block}

\section*{{8. Empirical results and what they support}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.96\linewidth]{{learning_curves.png}}
\caption{{Optimization diagnostics. Flow/diffusion panels report regression MSE;
the VAE panel reports reconstruction cost plus KL. Held-out objectives appear in
the table and are not sample-quality metrics.}}
\end{{figure}}

\begin{{table}}[H]
\centering
\scriptsize
\begin{{tabular}}{{llrrrrrrr}}
\toprule
Family & Impl. & Fr\'echet & MMD$^2$ & Prec. & Recall & Entropy & Classes & NN dist. \\
\midrule
{_mean_metric_rows(metrics)}
\bottomrule
\end{{tabular}}
\caption{{Means over {run_metadata['evaluation_replicates']} paired sampling seeds,
{run_metadata['generated_per_replicate']} samples per seed. Metrics are complementary,
not a composite leaderboard.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\linewidth]{{distribution_metrics.png}}
\caption{{A single metric can reward one failure mode and hide another. Bars are
means and error bars are standard deviations across sample-generation seeds.}}
\end{{figure}}

The precision/recall split follows the principle that fidelity and coverage are
separate axes \cite{{sajjadi2018precision}}. MMD detects broader distributional
differences; class entropy and class coverage diagnose mode omission through a
fixed digit probe; median nearest-training distance is a basic copying diagnostic.
None replaces conditional, human, subgroup, or application-specific evaluation.

\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{llrrr}}
\toprule
Family & Metric & MLP mean & Linear mean & MLP--linear [95\% interval] \\
\midrule
{_comparison_rows(comparison)}
\bottomrule
\end{{tabular}}
\caption{{Paired bootstrap across generation seeds. Intervals quantify Monte
Carlo sampling variation conditional on one fitted model; they exclude retraining
uncertainty and must not be read as a population confidence interval.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\linewidth]{{decoded_sample_grid.png}}
\caption{{Decoded examples support visual debugging, not quantitative claims.
Every row uses the same training-fitted PCA inverse and display scale.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.86\linewidth]{{flow_trajectories.png}}
\caption{{First two latent coordinates of PyTorch MLP flow-matching trajectories.
The full eight-dimensional integration is used for metrics.}}
\end{{figure}}

\section*{{9. Numerical cost and sampler sensitivity}}

\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{lrrrrr}}
\toprule
Family & Steps & Network evals. & Batch sec. & Fr\'echet & MMD$^2$ \\
\midrule
{_tradeoff_rows(tradeoff)}
\bottomrule
\end{{tabular}}
\caption{{PyTorch samplers on one fixed seed. RK4 uses four vector-field
evaluations per step; deterministic diffusion uses one noise prediction per step;
the VAE uses one decoder evaluation.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\linewidth]{{sampling_tradeoff.png}}
\caption{{Step count is both a numerical-analysis parameter and a deployment-cost
parameter. A fair comparison reports network evaluations and quality together.}}
\end{{figure}}

\section*{{10. Assessment plan and companion homework}}

The student-facing assignment uses Fashion-MNIST, not digits. Students implement
or extend VAE, flow, and diffusion objectives, verify analytic identities, define
PyTorch \code{{nn.Module}} models, train them with autograd and Adam, compare a
flow capacity ablation, contrast one-pass and iterative samplers, and write a
short model card. The
assignment is scoped so a CPU-only reduced subset is valid; an optional extension
may replace the PCA/MLP backbone with a convolutional PyTorch architecture without
changing the mathematical tests.

Suggested grading weights are: derivations 20\%, implementation correctness and
tests 25\%, controlled experiment 25\%, evaluation/uncertainty 20\%, and
reproducibility/communication 10\%. Solutions specify exact derivations where
available and explicit criteria for open-ended design judgments.

\section*{{11. Limitations and capstone extensions}}

\begin{{itemize}}
\item \textbf{{Capacity.}} PCA plus small MLPs cannot represent image detail that
modern convolutional or transformer backbones capture.
\item \textbf{{Conditioning scope.}} The controlled 8-by-8 benchmark is
unconditional; the 28-by-28 MNIST demonstration is label-conditional. Neither
tests free-form conditioning or classifier-free guidance.
\item \textbf{{One training seed.}} Paired generation seeds quantify sampler
variation, not optimization variability. A research claim requires repeated fits.
\item \textbf{{Metric validity.}} The benchmark lacks perceptual features, human
evaluation, calibrated privacy attacks, subgroup analysis, and downstream utility.
\item \textbf{{Path and schedule.}} Independent flow coupling and a linear beta
schedule are transparent baselines, not universally optimal choices.
\item \textbf{{Variational family.}} Diagonal Gaussian posteriors, fixed decoder
likelihoods, and $\beta=1$ are transparent VAE choices, not universally optimal;
the two-moons prior mismatch visibly smooths between modes.
\item \textbf{{Density.}} The benchmark does not estimate CNF likelihood through
the divergence integral; the affine-flow unit separately validates exact density.
\end{{itemize}}

Strong capstone extensions include minibatch OT flow matching, conditional
generation, probability-flow ODE comparison, latent diffusion, score calibration,
Schr\"odinger bridges, discrete/categorical flows, likelihood estimation with
Hutchinson traces, and a domain-specific evaluation protocol. LLMs can be added
later as a separate autoregressive sequence-modeling module.

\section*{{12. Reproducibility map}}

The build has one numerical source of truth and stable topic-scoped outputs:
\begin{{itemize}}
\item experiment: \path{{experiments/generative/run_benchmark.py}};
\item visual experiments: \path{{experiments/generative/run_visual_examples.py}};
\item reusable code: \path{{src/math550/topics/generative}};
\item mathematical tests: \path{{tests/generative}};
\item tables and figures: \path{{output/generative/analysis}} and
\path{{output/generative/figures}}; and
\item classroom artifacts: \path{{slides/generative}} and
\path{{homework/generative}}.
\end{{itemize}}
The JSON run configuration records the split, PCA, target paths, schedules,
architecture, PyTorch version/device, seeds, sample counts, and metric scope.
The separate \path{{visual_examples.json}} records MNIST checksums, the exact
two-moons Code 1 settings, VAE likelihood/posterior choices, training components,
sampler settings, probes, and caveats.

\begin{{thebibliography}}{{99}}
\bibitem{{lipman2024guide}} Y. Lipman et al. \emph{{Flow Matching Guide and Code}}.
arXiv:2412.06264, 2024. \url{{https://arxiv.org/abs/2412.06264}}.
\bibitem{{lipman2023flow}} Y. Lipman et al. \emph{{Flow Matching for Generative
Modeling}}. ICLR, 2023. \url{{https://arxiv.org/abs/2210.02747}}.
\bibitem{{chen2018neural}} R. T. Q. Chen et al. \emph{{Neural Ordinary
Differential Equations}}. NeurIPS, 2018.
\url{{https://proceedings.neurips.cc/paper/7892-neural-ordinary-differential-equations}}.
\bibitem{{dinh2017realnvp}} L. Dinh, J. Sohl-Dickstein, and S. Bengio.
\emph{{Density Estimation Using Real NVP}}. ICLR, 2017.
\url{{https://arxiv.org/abs/1605.08803}}.
\bibitem{{ho2020ddpm}} J. Ho, A. Jain, and P. Abbeel. \emph{{Denoising Diffusion
Probabilistic Models}}. NeurIPS, 2020.
\url{{https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html}}.
\bibitem{{song2021sde}} Y. Song et al. \emph{{Score-Based Generative Modeling
through Stochastic Differential Equations}}. ICLR, 2021.
\url{{https://openreview.net/forum?id=PxTIG12RRHS}}.
\bibitem{{kingma2014vae}} D. P. Kingma and M. Welling. \emph{{Auto-Encoding
Variational Bayes}}. ICLR, 2014. \url{{https://arxiv.org/abs/1312.6114}}.
\bibitem{{goodfellow2014gan}} I. Goodfellow et al. \emph{{Generative Adversarial
Nets}}. NeurIPS, 2014.
\url{{https://papers.nips.cc/paper/5423-generative-adversarial-nets}}.
\bibitem{{oord2016pixel}} A. van den Oord, N. Kalchbrenner, and K. Kavukcuoglu.
\emph{{Pixel Recurrent Neural Networks}}. ICML, 2016.
\url{{https://proceedings.mlr.press/v48/oord16.html}}.
\bibitem{{sajjadi2018precision}} M. S. M. Sajjadi et al. \emph{{Assessing
Generative Models via Precision and Recall}}. NeurIPS, 2018.
\url{{https://proceedings.neurips.cc/paper/2018/hash/f7696a9b362ac5a51c3dc8f098b73923-Abstract.html}}.
\bibitem{{uciDigits}} E. Alpaydin and C. Kaynak. \emph{{Optical Recognition of
Handwritten Digits}} [Dataset]. UCI Machine Learning Repository, 1998.
\url{{https://doi.org/10.24432/C50P49}}.
\bibitem{{mnistOriginal}} Y. LeCun, C. Cortes, and C. J. C. Burges.
\emph{{The MNIST Database of Handwritten Digits}} [Dataset].
\url{{https://yann.lecun.org/exdb/mnist/}}.
\bibitem{{pytorchDocs}} PyTorch Contributors. \emph{{PyTorch Documentation}}.
\url{{https://docs.pytorch.org/docs/stable/}}.
\end{{thebibliography}}

\end{{document}}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tex, encoding="utf-8")


def compile_latex_report(tex_path: Path) -> Path:
    """Compile the report with latexmk and return the stable PDF path."""

    command = [
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-cd",
        str(tex_path.resolve()),
    ]
    environment = os.environ.copy()
    environment.update({"LC_ALL": "C", "LANG": "C"})
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, env=environment
    )
    if result.returncode != 0:
        raise RuntimeError(
            "LaTeX compilation failed:\n" + result.stdout[-4000:] + result.stderr[-2000:]
        )
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise RuntimeError("LaTeX completed without producing a readable PDF")
    return pdf_path
