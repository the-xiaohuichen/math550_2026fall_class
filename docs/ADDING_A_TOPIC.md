# Adding a Teaching Topic

Use this checklist when adding a topic such as generative models, causal
inference, time series, graphical models, or deep learning.

## 1. Create the vertical slice

For a generative-model topic, start with:

```text
src/math550/topics/generative/
  __init__.py
  data.py
  validation.py
  evaluation.py
  plotting.py
  reporting.py
  models/
    __init__.py
    ...
experiments/generative/
  __init__.py
  config.py
  diagnostics.py
  run_benchmark.py
tests/generative/
  __init__.py
  test_models.py
docs/topics/generative.md
notes/generative/
  lecture_notes.tex
  lecture_notes.pdf
homework/generative/
  homework.tex
  homework.pdf
  README_data.md
  solutions.tex
  solutions.pdf
  solutions/
    validate_solutions.py
slides/generative/
  pdf/generative_models_teaching.pdf
  assets/manifest.json
  source/
    build_generative_deck.mjs
    import_to_keynote.applescript
    generative_models_teaching.pptx
```

Create only files justified by the topic. A small topic does not need empty
modules simply to match the example.

## 2. Define a public API

Export stable teaching classes and public data loaders from
`math550.topics.<topic>.__init__`. Keep tree nodes, training-state records, and
other implementation details private. Use consistent `fit`, `predict`,
`predict_proba`, `transform`, `sample`, or `score_samples` interfaces as
appropriate to the mathematical task.

## 3. Add topic dependencies

Add a named extra in `pyproject.toml`; for example:

```toml
[project.optional-dependencies]
generative = [
  "torch>=...",
  "torchvision>=...",
]
```

Then run `uv lock` and verify with `uv sync --locked --extra generative`. Avoid
putting a large topic-specific framework in the base dependency list.

## 4. Build the controlled comparison

In `experiments/<topic>/config.py`, define an immutable run configuration. For a
classical model, add paired scratch/reference specifications and record matched
settings and known implementation differences. For a neural model, use PyTorch
and specify a meaningful baseline or architecture ablation rather than a duplicate
manual-backprop implementation. The runner should receive one configuration
object, write topic-scoped artifacts, and never tune on the held-out test set.

## 5. Test the mathematics

Include tests for:

- output shapes, finiteness, ranges, and normalization constraints;
- a finite-difference gradient or known analytic case for classical code, or
  PyTorch module/shape/autograd checks for neural code;
- decreasing objectives when the algorithm should be monotone;
- deterministic behavior under a fixed seed;
- end-to-end performance on a public benchmark;
- quantitative agreement with a standard-library reference for classical code,
  or a controlled PyTorch baseline/ablation for neural code.

Synthetic tests supplement but do not replace the public benchmark.

## 6. Produce topic-scoped reports, notes, slides, and homework

Use:

```text
output/generative/analysis/
output/generative/figures/
output/pdf/generative_models_report.tex
output/pdf/generative_models_report.pdf
slides/generative/pdf/generative_models_teaching.pdf
slides/generative/source/generative_models_teaching.pptx
slides/generative/assets/manifest.json
```

The report must contain the model objective, optimization, data protocol,
classical reference gap or neural baseline/ablation, convergence behavior,
figures, interpretation, limitations, and reusable next steps. Compile and
visually inspect every PDF page.

The slide deck must complement the report with a classroom narrative. Export the
PDF from the native Keynote deck when reliable, reopen the `.key`, visually
inspect every PDF page, and run a structural verification target. If Keynote
cannot import in the available automation session, retain the editable PPTX and
export an appearance-preserving PDF from verified slide renders, documenting the
fallback. Verify that equations and reported numbers match the analysis tables.
Keep a topic-specific asset manifest and regeneration source.

Lecture notes must provide the detailed student-facing derivations,
implementation map, diagnostics, interpretation, limitations, and references.
They should stand alone without the slide deck. Homework should normally use a
different public dataset from the classroom demonstration, test transfer across
mathematics, implementation, verification, empirical analysis, and
interpretation, and keep complete instructor solutions separate. Run
computational solutions in the locked `uv` environment and retain a
machine-readable validation record.

## 7. Add commands and documentation

Add explicit Make targets such as `benchmark-generative` and
`quick-generative`. Update the root topic table and write
`docs/topics/generative.md` with the public API, dataset, references, commands,
outputs, and suggested classroom sequence.

## Completion checklist

- [ ] Classical scratch/reference pair, or PyTorch neural model with explicit loop
- [ ] Appropriate standard reference, baseline, or architecture ablation
- [ ] Immutable configuration and fixed random seed
- [ ] Public benchmark and source/license metadata
- [ ] Automated mathematical and interface tests
- [ ] Quantified head-to-head gap and investigated discrepancies
- [ ] Topic-scoped CSV/JSON results and figures
- [ ] LaTeX source and visually verified PDF
- [ ] Detailed lecture-note source and visually verified PDF
- [ ] Editable source deck and verified PDF; native Keynote when supported
- [ ] Slide asset provenance and reproducible generation source
- [ ] Every slide visually checked against the final export
- [ ] Locked `uv` dependencies and documented run command
- [ ] Different-dataset homework, data guide, separate solutions, and validated computations
