"""Public benchmark data loaders and dataset metadata."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
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
    class_labels: tuple[str, ...]
    positive_class: str | None


WDBC_METADATA = DatasetMetadata(
    name="Breast Cancer Wisconsin (Diagnostic)",
    source_url=(
        "https://archive.ics.uci.edu/dataset/17/"
        "breast-cancer-wisconsin-diagnostic"
    ),
    doi="10.24432/C5DW2B",
    license="CC BY 4.0",
    task="Binary classification from 30 continuous cell-nucleus measurements",
    class_labels=("benign", "malignant"),
    positive_class="malignant",
)

DRY_BEAN_METADATA = DatasetMetadata(
    name="Dry Bean",
    source_url="https://archive.ics.uci.edu/dataset/602/dry+bean+dataset",
    doi="10.24432/C50S4B",
    license="CC BY 4.0",
    task="Seven-class classification from 16 image-derived shape measurements",
    class_labels=(
        "BARBUNYA",
        "BOMBAY",
        "CALI",
        "DERMASON",
        "HOROZ",
        "SEKER",
        "SIRA",
    ),
    positive_class=None,
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

DRY_BEAN_FEATURE_NAMES = (
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRatio",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "Roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
)

DRY_BEAN_CLASS_COUNTS = {
    "BARBUNYA": 1322,
    "BOMBAY": 522,
    "CALI": 1630,
    "DERMASON": 3546,
    "HOROZ": 1928,
    "SEKER": 2027,
    "SIRA": 2636,
}

PROJECT_DRY_BEAN_PATH = Path(__file__).resolve().parents[4].joinpath(
    "data", "tabular", "dry_bean", "Dry_Bean_Dataset.arff"
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


def load_dry_bean(
    path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, DatasetMetadata]:
    """Load and validate the preserved seven-class UCI Dry Bean ARFF file."""

    requested_path = Path(path) if path is not None else PROJECT_DRY_BEAN_PATH
    if not requested_path.is_file():
        raise FileNotFoundError(requested_path)

    lines = requested_path.read_text(encoding="utf-8").splitlines()
    try:
        data_start = next(
            index
            for index, line in enumerate(lines)
            if line.strip().lower() == "@data"
        )
    except StopIteration as error:
        raise ValueError(
            "Dry Bean ARFF source does not contain an @DATA marker"
        ) from error

    data_lines = [
        line
        for line in lines[data_start + 1 :]
        if line.strip() and not line.lstrip().startswith("%")
    ]
    columns = (*DRY_BEAN_FEATURE_NAMES, "Class")
    raw = pd.read_csv(StringIO("\n".join(data_lines)), names=columns)
    predictors = raw.loc[:, list(DRY_BEAN_FEATURE_NAMES)].astype(float)
    target = raw["Class"].astype("string").rename("class")

    if predictors.shape != (13611, 16):
        raise ValueError(f"unexpected Dry Bean shape: {predictors.shape}")
    if not np.isfinite(predictors.to_numpy()).all():
        raise ValueError("Dry Bean predictors contain NaN or infinity")
    observed_counts = target.value_counts().sort_index().to_dict()
    if observed_counts != DRY_BEAN_CLASS_COUNTS:
        raise ValueError(f"unexpected Dry Bean class counts: {observed_counts}")
    return predictors, target, DRY_BEAN_METADATA
