"""Public benchmark data loaders and dataset metadata."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer


@dataclass(frozen=True)
class DatasetMetadata:
    """Documentation needed to reproduce and interpret a benchmark."""

    name: str
    source_url: str
    doi: str
    license: str
    task: str
    positive_class: str


WDBC_METADATA = DatasetMetadata(
    name="Breast Cancer Wisconsin (Diagnostic)",
    source_url=(
        "https://archive.ics.uci.edu/dataset/17/"
        "breast-cancer-wisconsin-diagnostic"
    ),
    doi="10.24432/C5DW2B",
    license="CC BY 4.0",
    task="Binary classification from 30 continuous cell-nucleus measurements",
    positive_class="malignant",
)


def load_wdbc() -> tuple[pd.DataFrame, pd.Series, DatasetMetadata]:
    """Load WDBC and encode malignant disease as the positive class.

    Scikit-learn distributes a local copy of this public UCI benchmark, so the
    experiment remains reproducible without a network request at runtime.
    """

    dataset = load_breast_cancer(as_frame=True)
    predictors = dataset.data.copy()
    target = (dataset.target == 0).astype("int8").rename("malignant")
    return predictors, target, WDBC_METADATA
