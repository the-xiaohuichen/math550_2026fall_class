# Repository Architecture

## Design goal

The repository should support many mathematically distinct teaching topics
without turning into one large model package. It uses vertical topic slices:
tabular classification owns its data contracts and metrics, while a future
generative-model topic can own image likelihoods, sampling diagnostics, and
tensor training loops without inheriting tabular assumptions.

## Dependency direction

```text
math550.core
    ^
    |
math550.topics.<topic>  <----  experiments.<topic>
    ^                              |
    |                              v
tests.<topic>                 output/<topic>
                                   |
                                   v
                              output/pdf
                                   |
                                   v
                            slides/<topic>
```

Rules:

1. `math550.core` contains only small, stable primitives useful to multiple
   topics. It must not import a topic.
2. `math550.topics.<topic>` contains reusable teaching code and exposes a small
   public API from its `__init__.py`.
3. `experiments.<topic>` owns references or ablations, run configuration,
   orchestration, and artifact writing. Reusable estimators must not import
   experiments.
4. `tests/<topic>` mirrors the source topic and verifies mathematical properties,
   interfaces, public-data behavior, and classical reference agreement or neural
   PyTorch/autograd behavior.
5. `output/<topic>/analysis` and `output/<topic>/figures` prevent artifact-name
   collisions. Stable PDFs remain in `output/pdf` with topic-specific names.
6. `slides/<topic>` owns an editable source deck, its verified PDF, an asset
   manifest, and regeneration source. Native Keynote is preferred on macOS;
   a portable PPTX plus an appearance-preserving PDF is an acceptable fallback
   when Keynote automation cannot import the deck. Slide assets may reference
   verified topic figures; they must not create a second numerical source of truth.
7. `notes/<topic>` owns detailed lecture notes that can stand alone without the
   slide deck. `homework/<topic>` owns a student handout, a different-dataset
   transfer task, separate instructor solutions, and computational validation.
   These artifacts may reuse verified figures and metrics but must not create
   conflicting numerical claims.
8. Reinforcement-learning topics keep environment adapters, replay data,
   Bellman targets, search, and self-play inside their topic slice. Environment
   packages may supply public dynamics, but high-level RL trainers must not own
   the teaching algorithm.

## File responsibilities

- `models/`: model definitions, objectives, and explicit training algorithms.
- `data.py`: public loaders, label conventions, and source/license metadata.
- `validation.py`: topic-specific input and output contracts.
- `evaluation.py`: metrics and statistical comparison procedures.
- `plotting.py`: pure figure builders that accept completed result tables.
- `reporting.py`: report source generation and PDF compilation.
- `experiments/<topic>/config.py`: immutable settings and reference/ablation factories.
- `experiments/<topic>/diagnostics.py`: analyses that are not estimator methods.
- `experiments/<topic>/run_benchmark.py`: orchestration and command-line entry.
- `slides/<topic>/source/`: editable interchange deck generation and, where
  supported, native Keynote import/export automation.
- `slides/<topic>/assets/manifest.json`: provenance for figures and numerical
  tables embedded in the teaching deck.
- `notes/<topic>/lecture_notes.tex`: durable derivations, implementation map,
  diagnostics, interpretation, and references.
- `homework/<topic>/`: student handout, public-data guide, instructor solutions,
  and a runnable solution validator.

## When code becomes shared

Do not move a helper into `math550.core` merely because it might be useful later.
Promote it only after a second topic needs the same mathematical behavior and
the shared name does not obscure the lesson. A stable sigmoid is shared; a
binary tabular feature-shape check is not.

## Dependencies

`pyproject.toml` keeps the minimal cross-topic dependency set in
`project.dependencies`. Each topic uses an optional dependency extra, such as
`tabular`. `uv.lock` resolves all extras reproducibly. This keeps installation and
CI costs proportional to the topics being used.

## Compatibility policy

Internal course code should import from `math550.topics.<topic>`. Experiment
modules are invoked with `python -m experiments.<topic>.run_benchmark`. Avoid
root-level topic scripts and generic output folders; both create naming conflicts
as the course grows.
