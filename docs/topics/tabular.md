# Tabular and Structured Data Topic

## Scope

This topic opens the black box behind common binary classifiers. Every model is
implemented with primitive NumPy operations and compared with a production
implementation under the same split and evaluation protocol.

| Family | Scratch implementation | Reference |
|---|---|---|
| Logistic regression | Stable cross-entropy and accelerated proximal gradient | `LogisticRegression` |
| L1/L2 regularization | Soft thresholding and smooth L2 gradient | `LogisticRegression` |
| Random forest | Weighted CART, bootstrap rows, random feature subsets | `RandomForestClassifier` |
| AdaBoost | Weighted decision stumps and exponential reweighting | `AdaBoostClassifier` |
| Gradient boosting | Trees fitted to `y - p` | `GradientBoostingClassifier` |
| XGBoost style | Exact second-order trees and regularized split gain | `XGBClassifier` |
| LightGBM style | Quantile histograms and leaf-wise growth | `LGBMClassifier` |

The XGBoost-style and LightGBM-style classes reproduce defining mathematical
mechanisms, not distributed training, GPU kernels, specialized categorical
handling, or every production safeguard.

## Benchmark

The benchmark is the public Breast Cancer Wisconsin (Diagnostic) dataset: 569
rows, 30 continuous predictors, and no missing values. The target is recoded to
`1 = malignant`. Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic),
DOI `10.24432/C5DW2B`, CC BY 4.0.

The official UCI source files are preserved under `data/tabular/`: WDBC for the
classroom benchmark and Wine for the transfer homework. Reproduce and verify
both downloads with `make download-tabular-data`; exact archive and file
SHA-256 digests are recorded in `data/tabular/download_manifest.json`.

## Public source API

Import estimators and the dataset loader from:

```python
from math550.topics.tabular import (
    ScratchLogisticRegression,
    ScratchRandomForestClassifier,
    load_wdbc,
)
```

Model implementation files are under
`src/math550/topics/tabular/models/`. Evaluation, data, plots, reports, and
validation remain in the parent topic package.

## Commands and outputs

```bash
uv sync --locked --extra tabular
make test
make benchmark-tabular
```

The full run writes numerical results to `output/tabular/analysis`, figures to
`output/tabular/figures`, and the LaTeX/PDF report to `output/pdf`. The
companion teaching deck is stored as native Keynote and Keynote-exported PDF
files under `slides/tabular/`; reproducible source and the portable interchange
deck are in `slides/tabular/source/`.

```bash
make verify-slides-tabular
```

Detailed lecture notes and the transfer assignment are topic-scoped:

```bash
make notes-tabular
make homework-tabular
make validate-homework-tabular
```

The Makefile discovers the repository-local or Homebrew `libomp` directory on
macOS; set `TABULAR_LIBOMP_DIR` explicitly for a nonstandard installation.

The 18-page notes in `notes/tabular/` derive the losses, gradients, proximal
updates, CART criteria, AdaBoost weights, Newton leaf values, split gain, and
histogram/leaf-wise design. They reuse the benchmark's five verified figures and
measured scratch/library gaps.

The homework in `homework/tabular/` uses the different public UCI Wine dataset:
178 observations, 13 chemical measurements, and all three original cultivar
classes. Student and instructor PDFs are separate. The instructor validator
checks a scratch multinomial softmax model, a transparent one-vs-rest adapter,
all eight model pairs, gradient and objective invariants, normalized three-class
probabilities, and a fixed stratified 133/45 split. In the validated run, every
absolute scratch/library macro one-vs-rest AUC gap was at most 0.0053.

For a larger optional scale-up exercise, `data/tabular/dry_bean/` preserves the
2020 UCI Dry Bean dataset: 13,611 rows, 16 numeric image-derived measurements,
and seven classes. The download manifest pins the official archive and source
file hashes; no full-model performance claim is attached until a separate
controlled experiment is run.

The verification target checks that the native `.key` package and the 15-page
PDF exist and are structurally readable. The deck itself was also reopened in
Keynote and every exported PDF page was visually inspected. Slides 4--5 contain
eight editable equations: the generator writes Office Math objects, and the
Keynote package preserves eight native equation resources after import.

## Suggested classroom sequence

1. Derive binary cross-entropy and inspect `ScratchLogisticRegression`.
2. Verify its gradient numerically and study monotone objective diagnostics.
3. Compare L1 soft thresholding with smooth L2 shrinkage.
4. Build a weighted CART stump and reuse it in AdaBoost.
5. Contrast independent bootstrap trees with sequential gradient trees.
6. Add Hessians and derive the XGBoost split-gain equation.
7. Replace exact thresholds with histograms and switch to leaf-wise growth.
8. Interpret AUC agreement separately from probability calibration.
