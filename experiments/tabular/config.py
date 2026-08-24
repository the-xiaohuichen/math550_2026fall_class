"""Configuration and matched model pairs for the tabular benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """One immutable, serializable configuration for a benchmark run."""

    output_root: Path = Path("output")
    n_bootstrap: int = 2_000
    quick: bool = False
    random_state: int = 550
    test_fraction: float = 0.20

    @property
    def analysis_dir(self) -> Path:
        return self.output_root / "tabular" / "analysis"

    @property
    def figure_dir(self) -> Path:
        return self.output_root / "tabular" / "figures"

    @property
    def pdf_dir(self) -> Path:
        return self.output_root / "pdf"

    def validate(self) -> None:
        if self.n_bootstrap < 100:
            raise ValueError("n_bootstrap must be at least 100")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must lie in (0, 1)")


@dataclass(frozen=True)
class PairSpec:
    """A controlled scratch/reference comparison and its teaching rationale."""

    family: str
    scratch: object
    library: object
    note: str
    matched_parameters: str
    known_differences: str


def build_pairs(n_train: int, config: ExperimentConfig) -> list[PairSpec]:
    """Construct every paired estimator from one shared run configuration."""

    # Heavy reference libraries are imported only when models are constructed.
    # Configuration inspection and documentation tooling therefore remain light.
    import numpy as np
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import (
        AdaBoostClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    from math550.topics.tabular import (
        ScratchAdaBoostClassifier,
        ScratchGradientBoostingClassifier,
        ScratchLightGBMClassifier,
        ScratchLogisticRegression,
        ScratchRandomForestClassifier,
        ScratchXGBoostClassifier,
    )

    forest_trees = 30 if config.quick else 100
    boost_trees = 40 if config.quick else 120
    gradient_trees = 50 if config.quick else 150
    lambda_matched_to_c1 = 1.0 / n_train

    def scratch_logistic(kind: str, strength: float) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ScratchLogisticRegression(
                        regularization=kind,
                        reg_strength=strength,
                        max_iter=10_000,
                        tol=1e-8,
                    ),
                ),
            ]
        )

    def library_logistic(l1_ratio: float, c_value: float, solver: str) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=c_value,
                        l1_ratio=l1_ratio,
                        solver=solver,
                        max_iter=10_000,
                        tol=1e-8,
                        random_state=config.random_state,
                    ),
                ),
            ]
        )

    return [
        PairSpec(
            "Logistic none",
            scratch_logistic("none", 0.0),
            library_logistic(0.0, np.inf, "lbfgs"),
            "Mean cross-entropy without a coefficient penalty.",
            "standardize; no penalty; max_iter=10,000; tol=1e-8",
            "Accelerated proximal gradient versus L-BFGS; separable data have no finite MLE.",
        ),
        PairSpec(
            "Logistic L1",
            scratch_logistic("l1", lambda_matched_to_c1),
            library_logistic(1.0, 1.0, "liblinear"),
            "Proximal soft-thresholding compared with liblinear L1.",
            "standardize; C=1; scratch lambda=1/n; max_iter=10,000; tol=1e-8",
            "FISTA soft thresholding versus liblinear coordinate descent and solver tolerances.",
        ),
        PairSpec(
            "Logistic L2",
            scratch_logistic("l2", lambda_matched_to_c1),
            library_logistic(0.0, 1.0, "lbfgs"),
            "Accelerated gradient compared with L-BFGS L2.",
            "standardize; C=1; scratch lambda=1/n; max_iter=10,000; tol=1e-8",
            "Accelerated gradient versus L-BFGS; objectives are matched after mean-loss scaling.",
        ),
        PairSpec(
            "Random forest",
            ScratchRandomForestClassifier(
                n_estimators=forest_trees,
                max_depth=5,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=config.random_state,
            ),
            RandomForestClassifier(
                n_estimators=forest_trees,
                max_depth=5,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=1,
                random_state=config.random_state,
            ),
            "Bootstrap trees with random feature subsets at every split.",
            f"trees={forest_trees}; depth=5; min_leaf=2; max_features=sqrt; bootstrap",
            "Independent random-number streams and tie-breaking create different individual trees.",
        ),
        PairSpec(
            "AdaBoost",
            ScratchAdaBoostClassifier(
                n_estimators=boost_trees,
                learning_rate=0.5,
                random_state=config.random_state,
            ),
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=boost_trees,
                learning_rate=0.5,
                random_state=config.random_state,
            ),
            "Weighted decision stumps with exponential-loss reweighting.",
            f"stumps={boost_trees}; learning_rate=0.5; depth=1",
            "The libraries use different score-to-probability conventions despite matching margins.",
        ),
        PairSpec(
            "Gradient boosting",
            ScratchGradientBoostingClassifier(
                n_estimators=gradient_trees,
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=5,
                random_state=config.random_state,
            ),
            GradientBoostingClassifier(
                n_estimators=gradient_trees,
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=5,
                random_state=config.random_state,
            ),
            "Trees fit sequentially to the negative binomial-loss gradient.",
            f"trees={gradient_trees}; learning_rate=0.05; depth=2; min_leaf=5",
            "The classroom model fits raw pseudo-residual means; sklearn applies optimized leaf updates.",
        ),
        PairSpec(
            "XGBoost style",
            ScratchXGBoostClassifier(
                n_estimators=boost_trees,
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=2,
                min_child_weight=1.0,
                reg_lambda=1.0,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=config.random_state,
            ),
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=boost_trees,
                learning_rate=0.05,
                max_depth=2,
                min_child_weight=1.0,
                reg_lambda=1.0,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                n_jobs=1,
                random_state=config.random_state,
            ),
            "Exact classroom Newton trees compared with production XGBoost hist.",
            f"trees={boost_trees}; eta=0.05; depth=2; min_child_weight=1; lambda=1; row/column=0.9",
            "Scratch uses exact thresholds; XGBoost uses its production histogram engine and safeguards.",
        ),
        PairSpec(
            "LightGBM style",
            ScratchLightGBMClassifier(
                n_estimators=boost_trees,
                learning_rate=0.05,
                num_leaves=7,
                max_depth=4,
                max_bins=32,
                min_samples_leaf=10,
                reg_lambda=1.0,
                random_state=config.random_state,
            ),
            LGBMClassifier(
                objective="binary",
                n_estimators=boost_trees,
                learning_rate=0.05,
                num_leaves=7,
                max_depth=4,
                max_bin=32,
                min_child_samples=10,
                reg_lambda=1.0,
                verbosity=-1,
                n_jobs=1,
                random_state=config.random_state,
            ),
            "Quantile histograms and leaf-wise growth compared with LightGBM.",
            f"trees={boost_trees}; eta=0.05; leaves=7; depth=4; bins=32; min_leaf=10; lambda=1",
            "Histogram construction, missing-value rules, tie-breaking, and production safeguards differ.",
        ),
    ]
