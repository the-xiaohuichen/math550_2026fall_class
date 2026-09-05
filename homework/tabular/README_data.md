# Homework datasets: UCI Wine and UCI Dry Bean

## Authoritative source and attribution

- Dataset: Wine
- Creator: M. Forina and collaborators, Institute of Pharmaceutical and Food
  Analysis and Technologies, Genoa, Italy
- Repository: UCI Machine Learning Repository
- Source: https://archive.ics.uci.edu/dataset/109/wine
- Official archive: https://archive.ics.uci.edu/static/public/109/wine.zip
- DOI: https://doi.org/10.24432/C5PC7J
- License listed by UCI: Creative Commons Attribution 4.0 International
- Local source file: `data/tabular/wine/wine.data`

The official UCI files are preserved under `data/tabular/wine/`, so the
assignment does not require a network request after checkout. Run
`make download-tabular-data` to download and checksum-verify the source archive
again. Students should cite the UCI record in reports.

## Task used in this homework

The original dataset contains 178 Italian wine samples, 13 continuous chemical
measurements, and three cultivar labels. This assignment preserves the original
three-class task:

- `y = 1`: original UCI cultivar class 1 (59 observations)
- `y = 2`: original UCI cultivar class 2 (71 observations)
- `y = 3`: original UCI cultivar class 3 (48 observations)

Classes 2 and 3 must remain separate. Scikit-learn's bundled copy relabels these
as 0, 1, and 2, but the downloaded UCI file and this assignment retain the
original 1, 2, and 3 labels.

The 13 predictors are alcohol, malic acid, ash, alcalinity of ash, magnesium,
total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color
intensity, hue, OD280/OD315 of diluted wines, and proline. The dataset has no
missing values. Several variables have very different physical units and
scales, so standardization is required for logistic regression and must be fit
on training observations only. Tree models may use the original scales.

The instructor validation and handout use a fixed three-class-stratified 80/20
train/test split with random seed 551. Penalty selection uses an additional
class-stratified 80/20 split of the outer training set, again with seed 551.
Accuracy, balanced accuracy, macro F1, per-class recall, confusion matrices,
and multiclass log loss expose different aspects of performance without
reducing the task to separate binary problems.

## Why this is a transfer task

The classroom demonstration uses cell-nucleus measurements from the Breast
Cancer Wisconsin Diagnostic dataset. Wine changes the scientific domain, sample
size, number and meaning of features, number of classes, class balance, and
measurement scales. Students must extend binary ideas with a joint softmax model
and native multiclass ensemble objectives; conclusions from the medical dataset
cannot simply be copied.

## Responsible use

This is a small chemical-analysis benchmark, not evidence about consumer taste,
safety, origin authenticity, or commercial quality. It contains no personal
data, but its geographic and historical sampling frame is narrow. A high test
score on this teaching split does not justify deployment on new regions,
vintages, instruments, or production processes. Any operational use would need
representative sampling, instrument-quality checks, drift monitoring, and a
clearly defined decision consequence.

## Reproducible loading

```python
from pathlib import Path

import pandas as pd

feature_names = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue",
    "od280_od315_of_diluted_wines", "proline",
]
path = Path("data/tabular/wine/wine.data")
raw = pd.read_csv(path, names=["uci_class", *feature_names])
X = raw[feature_names]
y = raw["uci_class"].astype(int)

assert X.shape == (178, 13)
assert y.value_counts().sort_index().to_dict() == {1: 59, 2: 71, 3: 48}
assert not X.isna().any().any()
```

Run the validated instructor computation with:

```bash
uv run --extra tabular python homework/tabular/solutions/validate_solutions.py
```

The script writes `homework/tabular/solutions/validation_results.json` and fails
if three-class shapes, normalized probabilities, optimization traces,
finite-difference gradients, or scratch/library agreement checks are invalid.

## Scale-up dataset: UCI Dry Bean

- Dataset: Dry Bean, donated to UCI on September 13, 2020
- Repository: UCI Machine Learning Repository, dataset 602
- Source: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- Official archive: https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip
- DOI: https://doi.org/10.24432/C50S4B
- License listed by UCI: Creative Commons Attribution 4.0 International
- Local source: `data/tabular/dry_bean/Dry_Bean_Dataset.arff`

Dry Bean contains 13,611 observations, 16 numeric measurements extracted from
segmented bean images, and seven registered bean-variety labels. The verified
class counts are:

- BARBUNYA: 1,322
- BOMBAY: 522
- CALI: 1,630
- DERMASON: 3,546
- HOROZ: 1,928
- SEKER: 2,027
- SIRA: 2,636

The majority-to-minority ratio is approximately 6.79, so macro F1, per-class
recall, and the confusion matrix are important complements to accuracy. UCI
reports no missing values, and the local checksum validator confirms 13,611
rows, 17 fields per row, finite numeric predictors, and the class counts above.

Load the local snapshot without a network request:

```python
from math550.topics.tabular import load_dry_bean

X, y, metadata = load_dry_bean()
assert X.shape == (13_611, 16)
assert y.nunique() == 7
```

The instructor split is class-stratified 80/20 with seed 552. Standardization
must be fit on training rows only. Dry Bean is about 76 times larger than Wine
and has more classes, making it useful for examining runtime, convergence,
class imbalance, and error concentration. It still comes from a particular
image-processing pipeline. A held-out score does not establish transfer to new
cameras, lighting, farms, cultivars, or harvests.
