# Generative Modeling Topic

## Scope and learning contract

This three-week module uses nine lecture/studio units to connect four views of
generative modeling: exact density models, latent-variable models, adversarial
models, and transport/denoising models. Variational autoencoders, flow matching,
and diffusion receive implementation depth; GANs, autoregressive models, and
normalizing flows provide additional model-selection context. Large language models are deliberately out
of scope.

By the end, students should be able to derive the main objectives, define and
train a small generator with standard PyTorch modules and autograd, separate training error from sampler error,
and defend a model choice with fidelity, coverage, memorization, compute, and
governance evidence.

## Nine-unit sequence

| Week | Unit | Main ideas | In-class evidence or studio |
|---|---:|---|---|
| 1 | 1 | Distribution learning; likelihood, ELBO, adversarial, and path objectives | Translate a model claim into an estimand and failure policy |
| 1 | 2 | ELBO, posterior gap, reparameterization, analytic Gaussian KL, likelihood choices | Train and audit a PyTorch VAE |
| 1 | 3 | Evaluation: latent Frechet, MMD, precision/recall, probes, nearest training distance | Design a held-out multi-metric audit |
| 2 | 4 | Change of variables, affine coupling, continuous normalizing flows | Execute inverse, density, and log-determinant checks |
| 2 | 5 | Continuity equation, probability paths, conditional flow matching | Derive the conditional velocity regression target |
| 2 | 6 | Euler/RK4 integration, path/coupling choices, flow-matching implementation | Stress path, solver, and network-evaluation budgets |
| 3 | 7 | Forward diffusion, schedules, epsilon/score/velocity targets | Verify corruption and oracle reconstruction identities |
| 3 | 8 | DDPM, DDIM, reverse SDE, probability-flow ODE | Compare stochastic and deterministic samplers |
| 3 | 9 | Unified path view, compute-quality curves, limitations, capstone decisions | Defend a model, metric suite, and failure response |

Recommended readings are Lipman et al.'s flow-matching paper and the detailed
flow-matching guide (arXiv:2412.06264), Ho et al. on DDPM, Song et al. on DDIM,
and Song et al. on score SDEs. The slide notes contain source links per slide.

## Three-example evidence ladder

The module now uses three complementary examples rather than asking one small
benchmark to do every teaching job.

1. **Two moons in 2D** follows Code 1 of the Flow Matching Guide closely:
   256 noisy moon observations, fresh standard-Gaussian sources, the straight
   interpolation path, a three-layer 64-unit ELU velocity network, 10,000 Adam
   updates, and an eight-step midpoint ODE sampler. The trajectory plot makes
   the learned transport field visible and supports direct comparison with the
   tutorial pseudocode. A two-dimensional Gaussian VAE is fitted to the same
   observations: its very small reconstruction MSE but poor prior-sample support
   exposes the posterior/prior gap and decoder smoothing directly.
2. **Original MNIST at 28-by-28** uses the official 60,000/10,000 split and a
   class-conditional compact U-Net-style PyTorch CNN. It turns Gaussian noise
   into recognizable native-resolution digits and exposes intermediate states,
   training curves, class-consistency probes, and nearest-reference distances.
   A class-conditional convolutional VAE uses the same official split, training
   budget, and label probe, while sampling with one decoder evaluation.
3. **Optical Digits at 8-by-8** remains the controlled audit: paired seeds,
   matched capacity ablations, multiple distribution metrics, sampling budgets,
   and decoded held-out comparisons. Its VAE adds an encoder, reconstruction
   audit, negative-ELBO decomposition, and a one-pass sampling operating point.

Together they separate mechanism visibility, qualitative usefulness, and
controlled quantitative evidence. The two new runs remain course-scale
demonstrations; they are not state-of-the-art image-generation claims.

## Controlled public benchmark and evidentiary limits

The controlled demonstration uses scikit-learn's public Optical Recognition of
Handwritten Digits data: 1,797 8-by-8 images, a fixed stratified 80/20 split,
and an eight-coordinate whitened PCA fitted on training rows only. The held-out
360 images define the target distribution. Eight generation seeds with 500
samples each compare a PyTorch Gaussian VAE, two-hidden-layer and linear
capacity variants for conditional flow matching and diffusion, and a
ten-component Gaussian-mixture baseline. Flow/diffusion capacity comparisons
use paired seeds and matched MSE/Adam/batch/epoch protocols; the VAE has the same
split and optimizer budget but an explicitly separate negative-ELBO scale.

The benchmark reports latent Frechet distance (not FID), RBF MMD-squared, k-NN
precision and recall, probe-label entropy and coverage, nearest-training distance,
runtime, and network-evaluation counts. Its role is to validate identities,
training logic, paired comparisons, and evaluation discipline. It does not
support claims about photorealism, broad architecture rankings, privacy, or
production image generation.

## Code and commands

The reusable code lives in `src/math550/topics/generative/`; the experiment
driver and matched architecture-ablation configuration live in
`experiments/generative/`. Neural layers, differentiation, losses, and Adam use
PyTorch; the course code keeps the probability paths, target construction,
training orchestration, samplers, and diagnostics explicit.

The benchmark accepts `--device auto`, `cuda`, `mps`, or `cpu`. `auto` prefers
CUDA, then Apple Metal, then CPU. Explicit unavailable accelerators fail with a
clear error, and the resolved device is recorded with the fitted-model metadata.

```bash
uv sync --locked --extra generative
make test-generative
make visual-generative
make benchmark-generative
make notes-generative
make verify-slides-generative
make validate-homework-generative
```

Use `make quick-generative` for reduced development runs. The full run
writes machine-readable results to `output/generative/analysis`, figures to
`output/generative/figures`, and the report to
`output/pdf/generative_models_report.pdf`.

## Teaching artifacts

- The 48-slide teaching deck and editable source are under `slides/generative/`.
- The 16-page rigorous benchmark report is `output/pdf/generative_models_report.pdf`.
- The 31-page standalone notes are `notes/generative/lecture_notes.tex` and
  `notes/generative/lecture_notes.pdf`.
- The 100-point student assignment and instructor solutions are under
  `homework/generative/`.
- The assignment transfers the workflow to the official Fashion-MNIST data and
  includes a checksum-verifying CPU reference validator.

The JavaScript authoring source and editable PPTX are reproducible deck sources.
The PPTX contains 55 native Office Math objects across the substantive formula
slides. Keynote imported all 48 slides, saved and reopened the native
`generative_models_teaching.key`, preserved 55 editable equation resources, and
exported the verified classroom PDF.
