"""Build an optional report-derived lecture-note scaffold.

The curated ``notes/generative/lecture_notes.tex`` is the canonical teaching
source and is compiled directly by ``make notes-generative``.  This helper writes
a separate scaffold so regenerating from the report cannot erase the expanded
derivations and instructor companion.
"""

from __future__ import annotations

import argparse
from pathlib import Path


TEACHING_APPENDIX = r"""
\clearpage
\section*{Teaching companion: derivations, checks, and discussion prompts}

The preceding sections are the evidence record. This companion turns that
record into a sequence of derivations and classroom decisions. Each subsection
has three parts: a mathematical object, a check that can fail, and a question
that forces students to interpret rather than merely execute code.

\subsection*{A. What a generator is actually learning}

Let $P_{\rm data}$ be the unknown population distribution and let
$x_1,\ldots,x_n$ be the observed sample. A generator defines a model
distribution $P_\theta$ from which new observations can be drawn. Three claims
must be kept separate:

\begin{enumerate}
\item \emph{fit}: $P_\theta$ matches aspects of the empirical distribution;
\item \emph{generalization}: those aspects also match unseen population draws;
\item \emph{utility}: generated observations are useful for a declared task.
\end{enumerate}

An exact likelihood can test the first two claims under a density model, but it
does not by itself establish perceptual fidelity or downstream utility. An
implicit generator may produce attractive samples while assigning no accessible
density. This is why the module treats likelihood, sample quality, coverage,
memorization, and compute as distinct axes.

\textbf{Check.} Ask whether every reported metric has a named reference set,
representation, sample count, and direction of improvement.

\textbf{Discuss.} Can two generators have identical likelihood and different
sample quality? Can they have identical sample quality and different coverage?

\subsection*{B. Autoregressive likelihood and latent-variable bounds}

The chain rule gives an exact factorization for any ordering:
\[
 p_\theta(x)=\prod_{j=1}^d p_\theta(x_j\mid x_{<j}),\qquad
 \log p_\theta(x)=\sum_{j=1}^d\log p_\theta(x_j\mid x_{<j}).
\]
Training can evaluate all observed conditionals in parallel when masking is
available, but ancestral sampling remains sequential in the chosen order. The
factorization is exact; the conditional model and its inductive biases are not.

For a latent-variable model, insert any distribution $q_\phi(z\mid x)$ and use
Jensen's inequality:
\begin{align*}
 \log p_\theta(x)
 &=\log\int q_\phi(z\mid x)
   \frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz\\
 &\geq \E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
 -D_{\rm KL}\!\left(q_\phi(z\mid x)\Vert p(z)\right).
\end{align*}
The gap is $D_{\rm KL}(q_\phi(z\mid x)\Vert p_\theta(z\mid x))$. A tight
bound therefore requires an expressive decoder \emph{and} an adequate
approximate posterior. The reparameterization
$z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon$ with
$\epsilon\sim\mathcal N(0,I)$ moves randomness outside the differentiable
network path.

\textbf{Check.} Verify that the KL term and reconstruction term use compatible
reductions. A sum over pixels and a mean over latent dimensions silently changes
their relative weight.

\textbf{Discuss.} Why can a powerful decoder ignore $z$? Which diagnostic would
distinguish posterior collapse from merely noisy latent coordinates?

\subsection*{B.1 The executable VAE contract}

The vector VAE uses a fixed-variance Gaussian decoder, so its per-example
negative ELBO (up to a parameter-independent constant) is
\[
 \frac{1}{2\sigma_x^2}\lVert x-\mu_\theta(z)\rVert_2^2
 +\frac12\sum_j\left[
 \mu_{\phi,j}(x)^2+\exp(\log\sigma_{\phi,j}^2(x))
 -1-\log\sigma_{\phi,j}^2(x)\right].
\]
The MNIST VAE replaces the Gaussian reconstruction cost with a 784-pixel
Bernoulli binary cross-entropy and conditions both encoder and decoder on the
digit label. Both are ordinary \code{torch.nn.Module} models; the course logic
keeps the posterior parameterization, reparameterized noise, reductions,
component-wise diagnostics, and prior sampler explicit.

\textbf{Check.} Assert a zero KL for $\mu=0,\log\sigma^2=0$, finite gradients for
both posterior heads, native output shape, and separate train/held-out curves for
reconstruction, KL, and total objective.

\textbf{Discuss.} Why can low reconstruction error coexist with poor prior
samples on a thin multimodal target?

\subsection*{C. Adversarial objectives and density ratios}

For fixed generator $G$, maximizing the original GAN value pointwise gives
\[
 D^*(x)=\frac{p_{\rm data}(x)}{p_{\rm data}(x)+p_G(x)}.
\]
Substitution yields a Jensen--Shannon divergence objective up to an additive
constant. This result assumes an optimal discriminator and overlapping support;
neither is guaranteed during finite alternating optimization. The common
non-saturating generator loss $-\E_z\log D(G(z))$ improves gradients but changes
the game followed by the actual iterates.

\textbf{Check.} Plot discriminator and generator losses together with held-out
coverage. A stable scalar loss is not evidence that modes are retained.

\textbf{Discuss.} What failure would be missed if a human evaluator sees only a
curated grid of high-quality samples?

\subsection*{D. Discrete flows and triangular Jacobians}

Suppose an affine coupling layer partitions $x=(x_a,x_b)$ and defines
\[
 y_a=x_a,\qquad y_b=x_b\odot\exp s_\theta(x_a)+t_\theta(x_a).
\]
The inverse is explicit:
\[
 x_a=y_a,\qquad
 x_b=(y_b-t_\theta(y_a))\odot\exp[-s_\theta(y_a)].
\]
The Jacobian is triangular, so
$\log|\det J|=\sum_j s_{\theta,j}(x_a)$. The neural networks $s_\theta$ and
$t_\theta$ are ordinary PyTorch modules; the architecture, not a hand-derived
neural optimizer, makes the determinant tractable.

\textbf{Check.} Numerically verify round-trip error and cancellation of forward
and inverse log determinants on random inputs before fitting any data.

\textbf{Discuss.} Why does exact invertibility restrict dimension-changing
operations? How do multiscale flows work around this constraint?

\subsection*{E. Continuous transport and the flow-matching identity}

Let $\phi_t$ solve $d\phi_t(x)/dt=u_t(\phi_t(x))$. Conservation of mass implies
\[
 \partial_t p_t(x)+\nabla\cdot[p_t(x)u_t(x)]=0.
\]
Along a trajectory this becomes
\[
 \frac{d}{dt}\log p_t(X_t)=-\nabla\cdot u_t(X_t).
\]
Continuous normalizing flows can integrate both state and density, but evaluating
the divergence can be expensive in high dimension.

Conditional flow matching avoids solving an ODE inside each training update.
Let $Z$ index a tractable conditional path with density $p_t(x\mid Z)$ and field
$u_t(x\mid Z)$. Define
\[
 p_t(x)=\int p_t(x\mid z)p(z)\,dz,
 \quad
 u_t(x)=\frac{\int u_t(x\mid z)p_t(x\mid z)p(z)\,dz}{p_t(x)}.
\]
Conditioning on $X_t=x$ shows that
$u_t(x)=\E[u_t(X_t\mid Z)\mid X_t=x]$. Squared-error regression therefore has
the same population minimizer whether it targets the unavailable marginal field
or sampled conditional fields. The conditional target is not an approximation
to a known marginal target; it is a stochastic regression target whose
conditional mean is the desired field.

For the classroom interpolation,
\[
 X_t=[1-(1-\sigma_{\min})t]X_0+tX_1,
 \qquad
 \dot X_t=X_1-(1-\sigma_{\min})X_0.
\]
The target is analytic. The learned model receives $[t,X_t]$ and predicts a
vector with the same dimension as $X_t$.

\textbf{Check.} Use finite differences to verify
$(X_{t+h}-X_t)/h\approx\dot X_t$, and test that an affine field reproduces a
known Gaussian transport before interpreting learned samples.

\textbf{Discuss.} Independent endpoint coupling is cheap. Why can an
optimal-transport coupling reduce path curvature, and what new estimation
dependence does minibatch coupling introduce?

\subsection*{E.1 The three-example evidence ladder}

The module deliberately assigns different jobs to three demonstrations. The
two-moons experiment reproduces the guide's page-7 Code 1 settings, makes
particle transport visible, and exposes VAE prior mismatch. Original 28-by-28
MNIST compares recognizable label-controlled flow and one-pass VAE generation.
The 8-by-8 PCA benchmark sacrifices sharpness so VAE, flow, diffusion, baselines,
seeds, metrics, and numerical budgets can be compared cheaply.
No one example is asked to prove mechanism, realism, and broad performance at
the same time.

\textbf{Check.} For every classroom claim, name which example supplies the
evidence and whether the evidence is visual, mathematical, or quantitative.

\textbf{Discuss.} Why would replacing the controlled PCA benchmark with only a
curated MNIST grid weaken the module, even though the grid is more visually
compelling?

\subsection*{F. ODE solvers are part of the model contract}

Euler's method uses
$x_{k+1}=x_k+h v_\theta(t_k,x_k)$ and has global error $O(h)$. Classical RK4
combines four field evaluations per step and has global error $O(h^4)$ under
regularity conditions. Consequently, ``30 steps'' is not a fair compute label
unless the solver and its number of network evaluations are also stated.

Three errors can be separated experimentally:
\begin{enumerate}
\item \emph{regression error}: the learned field misses its conditional target;
\item \emph{path error}: the chosen probability path is difficult or undesirable;
\item \emph{integration error}: a finite-step solver misses the learned flow.
\end{enumerate}
Holding two components fixed while varying the third turns vague sample-quality
complaints into a diagnostic experiment.

\textbf{Check.} Halve the step size and compare endpoints. If samples change
substantially while training loss is fixed, numerical error is material.

\subsection*{G. Diffusion objectives from the forward marginal}

Let $\alpha_t=1-\beta_t$ and
$\bar\alpha_t=\prod_{s=1}^t\alpha_s$. Repeated Gaussian transitions give
\[
 x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
 \qquad \epsilon\sim\mathcal N(0,I).
\]
This closed form permits uniform sampling of time indices during training. A
noise-prediction network minimizes
\[
 \E\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|_2^2.
\]
Equivalent parameterizations predict $x_0$, the score, or a velocity-like
combination, but their weighting and numerical behavior differ across signal-to-
noise ratios. The score identity is
\[
 \nabla_{x_t}\log q(x_t\mid x_0)
 =-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.
\]

For deterministic DDIM-style sampling, estimate
\[
 \hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,epsilon_\theta(x_t,t)}
 {\sqrt{\bar\alpha_t}}
\]
and combine $\hat x_0$ with the predicted noise at the next selected time. Skipping
times changes the numerical path; it does not retrain the network.

\textbf{Check.} Substitute the true noise into the reconstruction formula and
verify recovery of $x_0$ to floating-point tolerance.

\textbf{Discuss.} Why do very low signal-to-noise times amplify error in an
$x_0$ reconstruction? How might loss weighting respond?

\subsection*{H. Reverse SDE and probability-flow ODE}

For a forward SDE
\[
 dX=f(X,t)\,dt+g(t)\,dW_t,
\]
the reverse-time dynamics depend on the score $s_t(x)=\nabla_x\log p_t(x)$:
\[
 dX=[f(X,t)-g(t)^2s_t(X)]\,dt+g(t)\,d\bar W_t,
\]
with time run backward. The probability-flow ODE
\[
 dX=[f(X,t)-\tfrac12g(t)^2s_t(X)]\,dt
\]
has the same marginal densities under the required regularity conditions. Same
marginals do not mean identical sample paths. Stochasticity, solver choice, and
conditioning can change individual outputs and compute.

\textbf{Check.} State whether a reported sampler is stochastic or deterministic,
which time grid it uses, and how many neural evaluations one sample requires.

\subsection*{I. PyTorch implementation and device discipline}

The module uses \code{torch.nn.Module}, standard linear/convolutional layers and
activations, \code{MSELoss}, binary cross-entropy, autograd, and PyTorch
optimizers.
The code does not reimplement neural layers or backpropagation. A command-line
device request can be \code{auto}, \code{cuda}, \code{mps}, or \code{cpu}.
The \code{auto} policy resolves CUDA first, then Apple Metal, then CPU; an
explicit unavailable accelerator raises an error. The fitted network, training
tensors, validation tensors, and prediction tensors share the resolved device,
which is recorded in the run metadata.

\begin{verbatim}
model = TorchVectorRegressor(
    hidden_sizes=(64, 64), model_kind="mlp", device="auto"
)
model.fit(train_features, train_targets,
          validation_features, validation_targets)
print(model.device_)
\end{verbatim}

Reproducibility includes more than a seed: report the PyTorch version, device,
data split, preprocessing fit scope, training-pair generator, optimizer, batch
size, epoch budget, and sampler grid. Accelerator kernels may still differ from
CPU arithmetic even under deterministic settings.

\textbf{Check.} Assert that every model parameter has the same device type as
the resolved device, and run a tiny CPU test in continuous integration.

\subsection*{J. Reading the public-data evidence}

The experiment uses 1,437 training images and 360 untouched test images. PCA is
fit on training rows only; its eight whitened coordinates retain 67.4\% of
training variance. This bottleneck makes the experiment inexpensive and exposes
metric mechanics, but it also places a hard ceiling on decoded detail.

The Gaussian mixture achieved the smallest mean latent Fr\'echet distance
(0.090). The flow-matching MLP achieved 0.133 and the matched linear ablation
0.095; the paired MLP-minus-linear difference was +0.038 with a 95\% interval
[+0.027,+0.050]. The MLP nevertheless gained about 0.321 in k-NN precision.
This apparent conflict is useful: Fr\'echet compares only Gaussian moments,
whereas local precision responds to sample neighborhoods. Neither number alone
declares a winner.

For diffusion, the MLP validation noise-prediction MSE (0.731) improved over the
linear ablation (0.895), but its mean latent Fr\'echet distance (0.116) remained
worse than the linear result (0.095). A better supervised objective can be
necessary without being sufficient for a better finite-step generator.

\textbf{Check.} Before interpreting any Fr\'echet number, name its feature space.
The reported quantity is not Inception FID.

\subsection*{K. A reusable evaluation protocol}

\begin{enumerate}
\item Freeze the train/test split and fit every representation on training data.
\item Use held-out real data as the distributional reference.
\item Run paired generation seeds so model differences share random conditions.
\item Report at least one global discrepancy, one local precision/coverage pair,
one memorization diagnostic, decoded samples, and compute.
\item Add confidence intervals for a declared comparison, not only standard
deviations for each model in isolation.
\item Inspect failures and disclose the representation, sample count, fitting
seeds, and sampler configuration.
\end{enumerate}

Nearest-training distance is a screen, not a privacy guarantee. A serious release
decision needs stronger membership-inference or exposure tests, subgroup analysis,
rights review, and a response plan for memorized or harmful samples.

\subsection*{L. Nine-unit instructor plan}

\begin{center}
\small
\begin{tabularx}{\linewidth}{p{0.07\linewidth}p{0.23\linewidth}X}
\toprule
Unit & Board derivation & Studio or exit evidence \\
\midrule
1 & Distribution claims and metric axes & Critique a one-metric model card \\
2 & ELBO and reparameterization & Diagnose a reduction or collapse bug \\
3 & GAN density ratio & Design a mode-coverage audit \\
4 & Coupling inverse and log determinant & Pass round-trip and log-det tests \\
5 & Continuity equation and CNF density & Separate state and density ODEs \\
6 & Conditional FM identity & Verify target velocity and solver convergence \\
7 & Diffusion forward marginal & Pass oracle reconstruction test \\
8 & Reverse SDE and probability-flow ODE & Compare stochastic and deterministic costs \\
9 & Paired benchmark interpretation & Defend a capstone model and failure policy \\
\bottomrule
\end{tabularx}
\end{center}

Each unit can be taught as 35 minutes of derivation, 15 minutes of code or
diagnostics, and 10 minutes of critique. Across a three-hour week, the remaining
time supports a longer evidence lab and a short break.

\subsection*{M. Glossary}

\begin{description}
\item[Coverage] Whether the generator represents the diversity or modes of the
target distribution.
\item[Explicit density] A model for which normalized $p_\theta(x)$ or a tractable
equivalent can be evaluated.
\item[Implicit model] A model that can sample but does not expose a tractable
normalized density.
\item[Probability path] A family $(p_t)_{t\in[0,1]}$ connecting a base and target
distribution.
\item[Pushforward] The distribution obtained by applying a map to a random
variable, written $f_\#P$.
\item[Score] The gradient $\nabla_x\log p_t(x)$.
\item[Vector field] A function assigning an instantaneous velocity to every time
and state.
\item[Network evaluation] One forward call to the learned neural field or
denoiser; a more comparable sampler-cost unit than raw step count.
\end{description}
"""


def build(source: Path, destination: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(
            f"benchmark report source not found: {source}; run make benchmark-generative"
        )
    text = source.read_text(encoding="utf-8")
    text = text.replace("\\documentclass[10pt]{article}", "\\documentclass[11pt]{article}")
    text = text.replace("\\usepackage[margin=0.72in]{geometry}", "\\usepackage[margin=0.82in]{geometry}")
    text = text.replace(
        r"\graphicspath{{../generative/figures/}}",
        r"\graphicspath{{../../output/generative/figures/}}",
    )
    text = text.replace(
        r"{\large A three-week graduate capstone module with reproducible evidence}",
        r"{\large Detailed standalone lecture notes with reproducible evidence}",
    )
    text = text.replace(
        r"\begin{thebibliography}{99}",
        TEACHING_APPENDIX + "\n\\clearpage\n\\begin{thebibliography}{99}",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("output/pdf/generative_models_report.tex"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notes/generative/lecture_notes_from_report.tex"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(build(args.source, args.output).resolve())
