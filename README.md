# MATH 550 Teaching Lab

This repository is a topic-oriented home for transparent machine-learning
implementations. Each topic keeps its models, data rules, evaluation logic,
figures, experiment configuration, tests, and report separate while reusing a
small shared numerical core.

The governing workflow is in [`.agent/skills/SKILL.md`](.agent/skills/SKILL.md).
Its central rule is preserved throughout the repository: classical pedagogical
implementations are tested against established references, while neural-network
architectures use PyTorch and are evaluated through meaningful baselines and
ablations rather than duplicate manual-backprop implementations.

## Repository layout

```text
src/math550/
  core/                         cross-topic numerical primitives
  topics/
    tabular/                    first independent teaching topic
      models/                   scratch estimators
      data.py                   public data and metadata
      validation.py             topic-specific input contracts
      evaluation.py             metrics and paired uncertainty
      plotting.py               figures
      reporting.py              LaTeX/PDF builder
    generative/                 VAEs, flows, flow matching, and diffusion
      models/                   PyTorch VAEs/MLPs/CNNs, affine flows, FM, DDIM
      data.py                   held-out digits protocol and PCA representation
      evaluation.py             fidelity, coverage, and memorization diagnostics
    gnn/                        message passing and geometric deep learning
      models.py                 explicit GCN, radial messages, EGNN, energy/forces
      data.py                   rMD17 aspirin split and provenance contract
    llm/                        bytes/BPE, GPT, SFT, reward modeling, and PPO
      tokenization.py           explicit byte and byte-level BPE tokenizers
      model.py                  causal attention and decoder-only GPT
      posttraining.py           reward, GAE, KL shaping, and PPO objectives
    reasoning/                  LoRA SFT, exact verification, self-consistency, GRPO
      data.py                   GSM8K/SVAMP contracts, prompts, response masks
      objectives.py             response log-probs, group advantages, clipped GRPO
    evaluation/                 capstone metrics, uncertainty, leakage, judge audits
      metrics.py                pass@k, calibration, and item-level accuracy
      statistics.py             Wilson intervals, paired bootstrap, exact McNemar
      contamination.py          normalized n-gram overlap screens
    reinforcement/              Bellman learning, DQN, PUCT, and self-play
      models/dqn.py             explicit replay and Double-DQN objective
      alpha_zero.py             policy/value network, search, self-play loop
experiments/
  tabular/
    config.py                   immutable run and model-pair configuration
    diagnostics.py              parameter and regularization checks
    run_benchmark.py            orchestration only
  generative/                   PyTorch architecture and sampler ablations
  gnn/                          aspirin energy/force benchmark and data preparation
  llm/                          data, pre-training, SFT, reward, PPO, generation
  reinforcement/               MinAtar DQN and Connect Four self-play runner
  reasoning/                   GSM8K SFT, evaluation, GRPO, measured summaries
tests/
  tabular/                      tests mirror the source topic
  llm/                          tokenizer, causality, masks, and PPO invariants
docs/
  ARCHITECTURE.md
  ADDING_A_TOPIC.md
  topics/tabular.md
  topics/llm.md
output/
  tabular/analysis/             CSV and JSON results
  tabular/figures/              report figures
  pdf/                          stable client-facing reports
notes/
  tabular/                      detailed standalone lecture notes
  generative/                   detailed standalone lecture notes
  llm/                          10-page derivation and implementation notes
  reinforcement/               RL derivations and MuZero bridge
  reasoning/                   SFT, scaling, verification, and GRPO notes
slides/
  tabular/
    tabular_models_teaching.key native classroom deck
    tabular_models_teaching.pdf PDF exported from Keynote
    assets/                     asset and provenance manifest
    source/                     reproducible deck and Keynote automation
  generative/
    generative_models_teaching.key native classroom deck
    generative_models_teaching.pdf PDF exported from Keynote
    assets/                     source and numerical provenance
    source/                     editable PPTX, generator, and export automation
  llm/                         editable 28-slide PPTX/PDF and source manifests
  reinforcement/               editable RL deck, PDF, assets, and source
  reasoning/                   editable 24-slide Keynote/PPTX/PDF and source
homework/
  tabular/                      Wine transfer assignment, solutions, validator
  generative/                   student PDF, instructor solutions, validator
  llm/                          Time Machine transfer lab, solutions, validator
  reinforcement/               Asterix transfer lab, solutions, validator
  reasoning/                   SVAMP transfer audit, solutions, validator
  gnn/                         Cora citation-graph transfer lab and validator
```

Reusable library code never depends on an experiment driver. Topic packages may
depend on `math550.core`, but the core never imports a topic. This keeps a future
topic such as generative modeling independent from tabular-model assumptions.

## Current topics

| Topic | Status | Guide |
|---|---|---|
| Tabular and structured data | Implemented and benchmarked | [`docs/topics/tabular.md`](docs/topics/tabular.md) |
| Generative models | Implemented and benchmarked | [`docs/topics/generative.md`](docs/topics/generative.md) |
| Graph neural networks and geometric deep learning | Implemented and benchmarked | [`docs/topics/gnn.md`](docs/topics/gnn.md) |
| Large language models from scratch | Implemented and smoke-validated | [`docs/topics/llm.md`](docs/topics/llm.md) |
| Reinforcement learning | Implemented and smoke-benchmarked | [`docs/topics/reinforcement.md`](docs/topics/reinforcement.md) |
| Reasoning models from an LLM | Implemented and MPS smoke-validated | [`docs/topics/reasoning.md`](docs/topics/reasoning.md) |
| AI evaluation and benchmarking capstone | Implemented and benchmarked | [`docs/topics/evaluation.md`](docs/topics/evaluation.md) |

## Environment and commands

Dependencies are locked with `uv`. Topic-specific heavy dependencies are grouped
as optional extras so future topics can add PyTorch or other stacks without
making every environment install them.

```bash
uv sync --locked --extra tabular --extra generative --extra gnn --extra llm --extra rl --extra reasoning --extra evaluation
make test
make benchmark-tabular
make visual-generative
make benchmark-generative
make notes-tabular
make notes-generative
make homework-tabular
make validate-homework-tabular
make verify-slides-tabular
make verify-slides-generative
make data-gnn
make test-gnn
make quick-gnn
make benchmark-gnn
make notes-gnn
make homework-gnn
make validate-homework-gnn
make verify-slides-gnn
make download-llm-data
make prepare-llm-data
make test-llm
make smoke-llm
make notes-llm
make homework-llm
make validate-homework-llm
make verify-slides-llm
make test-reinforcement
make quick-reinforcement
make demo-reinforcement
make notes-reinforcement
make homework-reinforcement
make validate-homework-reinforcement
make verify-slides-reinforcement
make download-reasoning-data
make test-reasoning
make smoke-reasoning
make sft-reasoning
make notes-reasoning
make homework-reasoning
make validate-homework-reasoning
make verify-slides-reasoning
make download-evaluation-data
make test-evaluation
make benchmark-evaluation
make train-math-capstone
make report-evaluation
make notes-evaluation
make homework-evaluation
make validate-homework-evaluation
make verify-slides-evaluation
```

Equivalent direct commands are:

```bash
uv run --extra tabular python -m unittest discover -s tests -v
uv run --extra tabular python -m experiments.tabular.run_benchmark
uv run --extra generative python -m experiments.generative.run_benchmark --device auto
uv run --extra llm python -m experiments.llm.pretrain --preset smoke --steps 30 --device auto
uv run --extra rl python -m experiments.reinforcement.run_benchmark --quick --device auto
uv run --extra reasoning python -m experiments.reasoning.sft --preset quick --device auto
```

Use `make quick-tabular` or `make quick-generative` for reduced development
runs, and `make quick-reinforcement` for the reduced RL pipeline. The verified
generative report combines VAE and tutorial-faithful flow-matching views of two
moons, VAE and flow generation on native 28-by-28 MNIST, and a controlled
Optical Digits benchmark spanning VAE, flow, diffusion, and a GMM. The
controlled comparisons use eight generation seeds with 500 samples per seed
and 2,000 paired-bootstrap resamples. The tabular report uses
the full configuration and 2,000 class-stratified paired-bootstrap samples. On
macOS, LightGBM and XGBoost may also require `brew install libomp`. The Makefile
discovers a repository-local or standard Homebrew installation; set
`TABULAR_LIBOMP_DIR` when the library lives elsewhere.

The LLM topic defaults to a cache-aware random sample of 16 English Project
Gutenberg texts with a whole-book train/validation split. `--book-limit N` has no
course-imposed maximum: it reuses eligible `raw/books/pg*.txt` files first and
downloads only enough randomly selected mirror archives to reach N. The random
seed, catalog hash, exact active selection, split, URLs, and file hashes are
recorded in the download manifest, which is the preprocessing source of truth.
Pinned Databricks Dolly 15k and Anthropic HH-RLHF files support the optional
post-training labs. The included earlier 16-book measurement and 30-step
public-book run verify the real-data training path but are explicitly not claims
about every new random sample or model quality; see the topic guide for exact
commands, responsible bulk-download sources, hardware presets, and warnings.
Pretraining remains single-device by default with CUDA, MPS, and CPU support;
an opt-in `torchrun`/DistributedDataParallel driver demonstrates one process per
CUDA GPU, synchronized gradients, distributed validation, and rank-0-only
checkpointing. Generation uses a fresh recorded sampling seed by default and
accepts `--seed` when an exactly repeatable sample is desired. It samples only
token IDs defined by the tokenizer, masks any intentionally padded model-output
rows, and validates the checkpoint's recorded tokenizer size and SHA-256 when
those fields are available.

The reinforcement topic uses MinAtar Breakout for consumer-scale DQN and PPO
labs, an explicit outcome-GRPO objective exercise, and Connect Four for complete
AlphaZero-style self-play. The included
50,000-transition Breakout replay is 0.78 MB compressed. The verified 5,000-step
combined M2 MPS smoke run took 35.40 seconds for DQN and 3.55 seconds for PPO;
its six-episode and four-game evaluations are explicitly not presented as
converged, multi-seed claims. An optional full
Atari Minari dataset has an exact download command in the topic guide.
`make demo-reinforcement` loads the learned DQN checkpoint and exports an
annotated episode GIF, static contact sheet, per-step decision trace, and
checkpoint provenance; the same driver supports PPO with `--algorithm ppo`.

The reasoning topic starts from revision-pinned Qwen2.5-0.5B-Instruct and adds
worked-solution LoRA SFT, exact-answer evaluation, self-consistency, and explicit
GRPO with verifiable reward. It includes all 7,473 GSM8K training rows, the
untouched 1,319-row test split, and 1,000 SVAMP transfer rows with hashes. The
checked 20-step M2 MPS run is pipeline evidence rather than a benchmark; see the
topic guide for presets, measured uncertainty, and the documented SVAMP label
inconsistency.

The final evaluation capstone turns those earlier model families into auditable
benchmark objects. It implements pass@k, Wilson intervals, paired bootstrap
comparisons, exact McNemar tests, calibration diagnostics, replayable model
adapters, lexical contamination screens, and swapped-order judge audits. The
measured classroom run re-scores frozen GSM8K generations and Fashion-MNIST
generator records; MATH-500 is held back as the distinct final-project transfer
set. The GPU-first `train-math-capstone` target reuses the pinned 0.5B Qwen LoRA
path with CUDA, then MPS, then CPU fallback under `--device auto`.

## Adding or maintaining topics

- Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) before changing boundaries.
- Follow [`docs/ADDING_A_TOPIC.md`](docs/ADDING_A_TOPIC.md) for the complete topic
  scaffold, dependency, experiment, test, and report checklist.
- Keep generated numerical outputs below `output/<topic>/` and final PDFs below
  `output/pdf/` with topic-specific filenames.
- Keep detailed student-facing explanations below `notes/<topic>/`, and keep
  transfer-oriented assignments plus separate instructor solutions below
  `homework/<topic>/`.
- Keep classroom decks below `slides/<topic>/`. Every substantive topic should
  include an editable source deck, a verified PDF, an asset manifest, and
  regeneration source. Native Keynote is preferred when the environment can
  import and reopen the deck without changing its appearance.
- Add shared code to `math550.core` only after at least two topics need the same
  abstraction; premature shared abstractions make teaching code harder to read.

The tabular classroom demonstration uses UCI Breast Cancer Wisconsin Diagnostic.
Its homework uses the different UCI Wine dataset and a fixed 133/45 split. The
validated homework computation reproduces eight scratch/library comparisons; all
absolute AUC gaps are at most 0.0056, while probability diagnostics reveal larger
AdaBoost and first-order gradient-boosting differences.
# math550_2026fall
