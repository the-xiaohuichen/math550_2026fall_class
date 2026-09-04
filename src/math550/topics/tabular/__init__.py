"""Public API for the tabular and structured-data teaching topic."""

from .data import (
    DRY_BEAN_METADATA,
    DatasetMetadata,
    WDBC_METADATA,
    load_dry_bean,
    load_wdbc,
)
from .models import (
    ScratchAdaBoostClassifier,
    ScratchDecisionTreeClassifier,
    ScratchGradientBoostingClassifier,
    ScratchLightGBMClassifier,
    ScratchLogisticRegression,
    ScratchRandomForestClassifier,
    ScratchRegressionTree,
    ScratchXGBoostClassifier,
)

__all__ = [
    "DatasetMetadata",
    "DRY_BEAN_METADATA",
    "ScratchAdaBoostClassifier",
    "ScratchDecisionTreeClassifier",
    "ScratchGradientBoostingClassifier",
    "ScratchLightGBMClassifier",
    "ScratchLogisticRegression",
    "ScratchRandomForestClassifier",
    "ScratchRegressionTree",
    "ScratchXGBoostClassifier",
    "WDBC_METADATA",
    "load_dry_bean",
    "load_wdbc",
]
