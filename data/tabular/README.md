# Tabular course datasets

This directory preserves exact UCI source files used by the classroom
benchmark, its transfer homework, and an optional larger scale-up exercise. The
files are small enough to keep in the project, so the activities do not depend
on a network request after checkout.

## Classroom dataset: WDBC

- Dataset: Breast Cancer Wisconsin (Diagnostic), UCI dataset 17
- Source record: <https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic>
- Download archive: <https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip>
- DOI: <https://doi.org/10.24432/C5DW2B>
- License listed by UCI: CC BY 4.0
- Local files: `wdbc/wdbc.data` and `wdbc/wdbc.names`
- Verified data shape: 569 rows, one ID, one diagnosis, and 30 predictors
- Diagnosis counts: 357 benign (`B`) and 212 malignant (`M`)

The course loader removes the ID and encodes `M` as the positive class.

## Homework dataset: Wine

- Dataset: Wine, UCI dataset 109
- Source record: <https://archive.ics.uci.edu/dataset/109/wine>
- Download archive: <https://archive.ics.uci.edu/static/public/109/wine.zip>
- DOI: <https://doi.org/10.24432/C5PC7J>
- License listed by UCI: CC BY 4.0
- Local files: `wine/wine.data`, `wine/wine.names`, and `wine/Index`
- Verified data shape: 178 rows, one class label, and 13 predictors
- Original UCI class counts: 59 in class 1, 71 in class 2, and 48 in class 3

The homework preserves all three original UCI class labels. Scikit-learn's
bundled loader renumbers the labels as 0, 1, and 2.

## Optional larger benchmark: Dry Bean

- Dataset: Dry Bean, UCI dataset 602
- Donated to UCI: September 13, 2020
- Source record: <https://archive.ics.uci.edu/dataset/602/dry+bean+dataset>
- Download archive: <https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip>
- DOI: <https://doi.org/10.24432/C50S4B>
- License listed by UCI: CC BY 4.0
- Local files: `dry_bean/Dry_Bean_Dataset.arff` and `dry_bean/Dry_Bean_Dataset.txt`
- Verified data shape: 13,611 rows, 16 numeric predictors, and one target
- Target: seven registered bean varieties
- Missing values reported by UCI and observed locally: none

The predictors are measurements extracted from segmented bean images. This is
a useful scale-up from Wine: it is about 76 times larger, has seven rather than
three classes, and exposes class imbalance and longer scratch-model runtimes.
It remains a curated benchmark, not evidence that a fitted model will transfer
to new cameras, acquisition conditions, farms, or harvests.

## Reproduce and audit

Run:

```bash
make download-tabular-data
```

`tools/download_tabular_data.py` downloads the three official archives into the
ignored `.cache/tabular/uci/` directory, verifies their pinned SHA-256 digests,
extracts only the documented source files, verifies every extracted digest,
and checks row widths and class counts. `download_manifest.json` is the
machine-readable provenance record.

These datasets are teaching benchmarks. WDBC does not support clinical use,
Wine does not establish product quality, authenticity, safety, or consumer
preference, and Dry Bean does not establish deployment performance under new
image-acquisition conditions. Retain the UCI attribution in redistributed
course materials.
