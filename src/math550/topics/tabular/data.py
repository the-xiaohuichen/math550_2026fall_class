"""Public benchmark data loaders and dataset metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


WDBC_FEATURE_NAMES = (
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
)

PROJECT_WDBC_PATH = Path(__file__).resolve().parents[4].joinpath(
    "data", "tabular", "wdbc", "wdbc.data"
)


def load_wdbc(
    path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, DatasetMetadata]:
    """Load WDBC and encode malignant disease as the positive class.

    The course repository preserves the official UCI ``wdbc.data`` file under
    ``data/tabular/wdbc``. A scikit-learn fallback keeps the installed package
    usable outside the repository without introducing a runtime download.
    """

    requested_path = Path(path) if path is not None else PROJECT_WDBC_PATH
    if requested_path.is_file():
        columns = ("id", "diagnosis", *WDBC_FEATURE_NAMES)
        raw = pd.read_csv(requested_path, names=columns)
        predictors = raw.loc[:, list(WDBC_FEATURE_NAMES)].astype(float)
        target = raw["diagnosis"].eq("M").astype("int8").rename("malignant")
        if predictors.shape != (569, 30):
            raise ValueError(f"unexpected WDBC shape: {predictors.shape}")
        if set(raw["diagnosis"]) != {"B", "M"}:
            raise ValueError("unexpected WDBC diagnosis labels")
        if predictors.isna().any().any():
            raise ValueError("WDBC predictors contain missing values")
        return predictors, target, WDBC_METADATA
    if path is not None:
        raise FileNotFoundError(requested_path)

    dataset = load_breast_cancer(as_frame=True)
    predictors = dataset.data.copy()
    target = (dataset.target == 0).astype("int8").rename("malignant")
    return predictors, target, WDBC_METADATA
