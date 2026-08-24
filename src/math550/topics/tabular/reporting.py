"""Generate the client-facing LaTeX source from verified benchmark outputs."""

from __future__ import annotations

from datetime import date
import os
from pathlib import Path
import subprocess

import pandas as pd


def _escape(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def write_latex_report(
    pair_table: pd.DataFrame,
    metric_table: pd.DataFrame,
    logistic_parameters: pd.DataFrame,
    model_configuration: pd.DataFrame,
    output_path: Path,
    n_train: int,
    n_test: int,
    n_features: int,
    n_bootstrap: int,
    random_state: int,
) -> None:
    """Write a polished report whose numbers come from the completed run."""

    consistent_count = int((pair_table["consistency"] == "consistent").sum())
    largest_gap = pair_table.loc[pair_table["auc_gap"].abs().idxmax()]
    smallest_gap = pair_table.loc[pair_table["auc_gap"].abs().idxmin()]
    scratch_best = metric_table[metric_table["implementation"] == "scratch"].loc[
        lambda frame: frame["test_roc_auc"].idxmax()
    ]
    library_best = metric_table[metric_table["implementation"] == "library"].loc[
        lambda frame: frame["test_roc_auc"].idxmax()
    ]
    l1_parameters = logistic_parameters.loc[
        logistic_parameters["family"] == "Logistic L1"
    ].iloc[0]
    adaboost_pair = pair_table.loc[pair_table["family"] == "AdaBoost"].iloc[0]
    gradient_pair = pair_table.loc[
        pair_table["family"] == "Gradient boosting"
    ].iloc[0]
    slowest_relative = pair_table.loc[pair_table["fit_time_ratio"].idxmax()]
    unregularized_metrics = metric_table.loc[
        metric_table["family"] == "Logistic none"
    ].set_index("implementation")

    paired_rows = []
    for _, row in pair_table.iterrows():
        status = (
            r"\textcolor{green!45!black}{consistent}"
            if row["consistency"] == "consistent"
            else r"\textcolor{red!70!black}{review}"
        )
        paired_rows.append(
            f"{_escape(row['family'])} & {row['scratch_roc_auc']:.3f} & "
            f"{row['library_roc_auc']:.3f} & {row['auc_gap']:+.3f} "
            f"[{row['auc_gap_ci_low']:+.3f}, {row['auc_gap_ci_high']:+.3f}] & "
            f"{row['label_agreement']:.3f} & {row['probability_mae']:.3f} & {status} \\\\"
        )

    performance_rows = []
    for _, row in metric_table.sort_values(["family", "implementation"]).iterrows():
        performance_rows.append(
            f"{_escape(row['family'])} & {_escape(row['implementation'])} & "
            f"{row['train_roc_auc']:.3f} & {row['test_roc_auc']:.3f} & "
            f"{row['test_accuracy']:.3f} & {row['test_recall_sensitivity']:.3f} & "
            f"{row['test_specificity']:.3f} & {row['test_log_loss']:.3f} & "
            f"{row['fit_seconds']:.3f} \\\\"
        )

    objective_rows = []
    for _, row in logistic_parameters.iterrows():
        converged = "yes" if bool(row["scratch_converged"]) else "no"
        objective_rows.append(
            f"{_escape(row['family'])} & {row['scratch_data_loss']:.4f} & "
            f"{row['scratch_regularization_loss']:.4f} & "
            f"{row['scratch_final_objective']:.4f} & "
            f"{row['library_objective']:.4f} & {row['objective_abs_gap']:.4f} & "
            f"{converged} \\\\"
        )

    configuration_rows = []
    for _, row in model_configuration.iterrows():
        configuration_rows.append(
            f"{_escape(row['family'])} & {_escape(row['matched_parameters'])} & "
            f"{_escape(row['known_differences'])} \\\\"
        )

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,graphicx,float,microtype,tabularx,xcolor}}
\usepackage[colorlinks=true,linkcolor=blue!45!black,urlcolor=blue!45!black]{{hyperref}}
\hypersetup{{
  pdftitle={{Modeling Tabular Data: From Mathematical Primitives to Production Libraries}},
  pdfauthor={{MATH 550 Capstone Course Development}},
  pdfsubject={{Scratch-versus-library benchmark of tabular classification models}}
}}
\graphicspath{{{{../tabular/figures/}}}}
\definecolor{{navy}}{{HTML}}{{1F4E78}}
\definecolor{{lightblue}}{{HTML}}{{D9EAF7}}
\pagestyle{{plain}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{4pt}}
\newcommand{{\code}}[1]{{\texttt{{#1}}}}

\begin{{document}}

\begin{{center}}
{{\LARGE\bfseries\color{{navy}} Modeling Tabular Data}}\\[4pt]
{{\Large From Mathematical Primitives to Production Libraries}}\\[10pt]
{{\large Client-facing benchmark and classroom implementation report}}\\[12pt]
Prepared for MATH 550 Capstone Course Development\\
Verified run: {date.today().isoformat()}
\end{{center}}

\vspace{{4pt}}
\colorbox{{lightblue}}{{\parbox{{0.96\linewidth}}{{
\textbf{{Executive finding.}} {consistent_count} of {len(pair_table)} paired
comparisons met the pre-declared ROC-AUC consistency rule. The smallest absolute
gap was {_escape(smallest_gap['family'])} ({smallest_gap['auc_gap']:+.3f}); the
largest was {_escape(largest_gap['family'])} ({largest_gap['auc_gap']:+.3f}).
The scratch implementations recover the predictive behavior of the standard
libraries while keeping the loss, split, weighting, and update equations visible.
The main caveat is probability calibration: AdaBoost has identical ranking but
probability MAE {adaboost_pair['probability_mae']:.3f} because the two versions
use different score-to-probability conventions.
}}}}

\section*{{1. Purpose and scope}}

This report validates reusable, from-scratch implementations of logistic
regression, random forests, AdaBoost, gradient boosting, XGBoost-style Newton
boosting, and LightGBM-style histogram leaf-wise boosting. Every scratch model is
evaluated head-to-head against its standard Python library counterpart on the same
held-out observations. The objective is algorithmic transparency, not a claim that
compact classroom code should replace mature production libraries.

The benchmark is the public UCI Breast Cancer Wisconsin Diagnostic dataset:
569 rows, {n_features} continuous predictors, and no missing cells. It is
distributed under CC BY 4.0 and loaded from scikit-learn's local copy. The
response is recoded so that $y=1$ denotes malignant disease.

The fixed class-stratified 80/20 split contains {n_train} training and {n_test}
test observations (random seed {random_state}). Standardization is fit on the
training observations and applied only to logistic models; tree models retain
the original numeric scale. The regularization path uses a further stratified
split within the training data. Hyperparameters are fixed before test evaluation.
All pairs use identical rows, labels, preprocessing policy, metrics, and seeds.
Primary uncertainty uses {n_bootstrap:,} class-stratified paired-bootstrap
resamples of the held-out observations.

\section*{{2. Mathematical implementation}}

\subsection*{{2.1 Logistic regression and regularization}}

For score $z_i=x_i^\top\beta+b$ and probability
$p_i=\sigma(z_i)=1/(1+e^{{-z_i}})$, the scratch class minimizes mean binary
cross-entropy
\[
  \mathcal{{L}}(\beta,b)=\frac{{1}}{{n}}\sum_{{i=1}}^n
  \left[\log(1+e^{{z_i}})-y_i z_i\right]+\Omega(\beta).
\]
The alternatives are $\Omega=0$, $\Omega=\lambda\lVert\beta\rVert_1$, and
$\Omega=\lambda\lVert\beta\rVert_2^2/2$. Training uses accelerated proximal
gradient descent. The L1 proximal step is soft thresholding;
$\mathrm{{prox}}_{{t\lambda}}(u)=\mathrm{{sign}}(u)\max(|u|-t\lambda,0)$.
The intercept is not penalized. The step size uses the global Lipschitz bound
$L=\lVert X\rVert_2^2/(4n)+\lambda_2$. A monotone restart and backtracking guard
handles FISTA overshoot. The implementation records the data-fit, penalty, and
total objective separately at every accepted iteration.

For L1, the scratch and library standardized coefficient vectors had correlation
{l1_parameters['coefficient_correlation']:.3f}, mean absolute difference
{l1_parameters['coefficient_mae']:.3f}, and support Jaccard index
{l1_parameters['support_jaccard']:.3f}. These diagnostics test parameter behavior,
not only final labels.

\subsection*{{2.2 Trees, bagging, and boosting}}

The scratch CART classifier enumerates midpoint thresholds and minimizes weighted
Gini impurity. Random forests bootstrap observations, select a random feature
subset at every node, and average tree probabilities. AdaBoost repeatedly fits a
weighted decision stump and updates observation weights using
$w_i\leftarrow w_i\exp(-\alpha y_i h_i)$.

Classical gradient boosting fits shallow regression trees to the negative
cross-entropy gradient $y-p$. The XGBoost-style implementation uses first and
second derivatives $g=p-y$ and $h=p(1-p)$; a candidate split has regularized gain
\[
\frac12\left[\frac{{G_L^2}}{{H_L+\lambda}}+
\frac{{G_R^2}}{{H_R+\lambda}}-
\frac{{(G_L+G_R)^2}}{{H_L+H_R+\lambda}}\right]-\gamma.
\]
The LightGBM-style implementation applies the same Newton objective after
quantile binning and grows the globally highest-gain leaf. It therefore explains
the core histogram and leaf-wise ideas without reproducing distributed systems,
GPU kernels, categorical optimizations, or every production safeguard.

\subsection*{{2.3 Reusable design and controlled reference mapping}}

All scratch estimators expose \code{{fit}}, \code{{predict}}, and
\code{{predict\_proba}}. Topic-specific contracts live in
\code{{validation.py}}, while stable sigmoid/logit operations live in the shared
\code{{math550.core}} package. The weighted classification tree is reused by
random forest and AdaBoost, and the regression tree by first-order gradient
boosting. The table distinguishes matched parameters from differences intrinsic
to a compact classroom implementation.

\begin{{table}}[H]
\centering
\scriptsize
\begin{{tabularx}}{{\linewidth}}{{p{{0.15\linewidth}}XX}}
\toprule
Family & Matched experimental settings & Intentional or unavoidable difference \\
\midrule
{chr(10).join(configuration_rows)}
\bottomrule
\end{{tabularx}}
\caption{{Configuration fixed before examining test results.}}
\end{{table}}

\section*{{3. Paired validation results}}

The primary comparison is test ROC AUC. The reported gap is scratch minus library;
its interval is a {n_bootstrap:,}-sample class-stratified paired bootstrap. A pair is called
\emph{{performance-consistent}} when the interval contains zero or the absolute
observed gap is at most 0.02. Label agreement and probability mean absolute error
(MAE) are complementary diagnostics.

\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{lrrrrrr}}
\toprule
Family & Scratch AUC & Library AUC & AUC gap [95\% CI] & Label agree. & Prob. MAE & Result \\
\midrule
{chr(10).join(paired_rows)}
\bottomrule
\end{{tabular}}
\caption{{Direct implementation comparison on the same {n_test} test observations.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\linewidth]{{head_to_head_auc.png}}
\caption{{Each point compares the scratch and standard library ROC AUC for one
model family. The diagonal represents exact parity.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.86\linewidth]{{paired_auc_gaps.png}}
\caption{{Paired-bootstrap uncertainty for scratch-minus-library ROC AUC.}}
\end{{figure}}

\section*{{4. Probability and optimization diagnostics}}

The strongest scratch AUC was {_escape(scratch_best['family'])}
({scratch_best['test_roc_auc']:.3f}); the strongest library AUC was
{_escape(library_best['family'])} ({library_best['test_roc_auc']:.3f}). These are
descriptive results across fixed configurations, not test-set model selection.
Observation-level probability plots reveal differences that AUC can hide: models
may rank cases similarly while assigning materially different confidence.

AdaBoost is the clearest example. Scratch and library versions have the same AUC
({adaboost_pair['scratch_roc_auc']:.3f}) and label agreement
{adaboost_pair['label_agreement']:.3f}, yet probability MAE is
{adaboost_pair['probability_mae']:.3f}. Their score-to-probability conventions
differ, producing test log loss {adaboost_pair['scratch_log_loss']:.3f} versus
{adaboost_pair['library_log_loss']:.3f}. This is not a ranking failure; it is a
calibration lesson.

Gradient boosting also shows a material probability discrepancy (MAE
{gradient_pair['probability_mae']:.3f}) despite an AUC gap of only
{gradient_pair['auc_gap']:+.3f}. The classroom version fits pseudo-residual means,
whereas scikit-learn applies optimized terminal-region updates. Thus AUC verifies
ranking consistency but does not establish calibration equivalence.

Unregularized logistic regression shows a second failure mode. The separable
training sample drives the cross-entropy objective toward zero without a finite
coefficient optimum. Both implementations retain AUC near
{unregularized_metrics.loc['scratch', 'test_roc_auc']:.3f}, but their test log losses
({unregularized_metrics.loc['scratch', 'test_log_loss']:.3f} scratch and
{unregularized_metrics.loc['library', 'test_log_loss']:.3f} library) are much worse
than the regularized fits because a small number of confident errors dominate
cross-entropy.

\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{lrrrrrr}}
\toprule
Model & Scratch data & Scratch penalty & Scratch total & Library total & Abs. gap & Converged \\
\midrule
{chr(10).join(objective_rows)}
\bottomrule
\end{{tabular}}
\caption{{Training objectives evaluated under the scratch mean-loss convention.
The unregularized case is separable, so a finite optimum and parameter parity are
not expected.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.96\linewidth]{{probability_agreement.png}}
\caption{{Paired test probabilities. Orange points are malignant observations;
blue points are benign.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.78\linewidth]{{logistic_optimization.png}}
\caption{{The explicit penalized-objective trace makes monotone convergence and
the effect of penalties inspectable.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.96\linewidth]{{regularization_effect.png}}
\caption{{Inner-validation loss, coefficient shrinkage, and L1 sparsity across
regularization strengths. The held-out test set is not used for this diagnostic.}}
\end{{figure}}

\section*{{5. Complete metric table}}

\begin{{table}}[H]
\centering
\scriptsize
\begin{{tabular}}{{llrrrrrrr}}
\toprule
Family & Impl. & Train AUC & Test AUC & Accuracy & Sens. & Spec. & Log loss & Fit s \\
\midrule
{chr(10).join(performance_rows)}
\bottomrule
\end{{tabular}}
\caption{{Test threshold metrics use a fixed probability cutoff of 0.5. Fit
times are descriptive single-process wall-clock measurements, not a formal
hardware benchmark.}}
\end{{table}}

\section*{{6. Interpretation and limitations}}

Small performance gaps support the correctness of the primitive implementations,
but exact equality is neither expected nor required. Random seeds, tie breaking,
stopping rules, probability transformations, and production-specific numerical
optimizations differ. XGBoost-style and LightGBM-style scratch models reproduce
the defining mathematical mechanisms, not the full engineering systems.

As expected, transparent Python/NumPy code is slower. The largest observed
scratch-to-library fit-time ratio was {_escape(slowest_relative['family'])} at
{slowest_relative['fit_time_ratio']:.1f}$\times$. This is an instructional
trade-off rather than a performance defect: production libraries use compiled,
vectorized, parallel, and memory-optimized kernels.

The test set has only {n_test} observations, so uncertainty is material. This
dataset is clean, numeric, and small; later course modules should add missing
values, categorical variables, high-cardinality features, calibration, and nested
cross-validation. Next steps are probability calibration, missing-value and
categorical splits, multiclass objectives, and profiling on a larger public
benchmark. These extensions can reuse the validation, tree, optimization, and
reporting components. Any importance measure would be predictive rather than
causal. Finally, this benchmark is educational and must not be used for clinical
decision-making.

\section*{{7. Reproducibility and code organization}}

Reusable source is grouped under \code{{src/math550/topics/tabular/}}, with
cross-topic numerics in \code{{src/math550/core/}}. One-off configuration,
diagnostics, and orchestration live in \code{{experiments/tabular/}}; tests mirror
the topic under \code{{tests/tabular/}}. \code{{pyproject.toml}} and
\code{{uv.lock}} specify the environment. Run \code{{uv sync}},
\code{{uv run python -m unittest discover -s tests -v}}, and
\code{{uv run python -m experiments.tabular.run\_benchmark}}. The run validates
probability contracts before metrics and retains topic-scoped tables, figures,
LaTeX, and PDF outputs.

\section*{{References}}

{{\footnotesize
\begin{{enumerate}}
\setlength{{\itemsep}}{{1pt}}
\item Wolberg, W., Mangasarian, O., Street, N., and Street, W. (1993).
Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository.
\url{{https://doi.org/10.24432/C5DW2B}}.
\item Friedman, J. H. (2001). Greedy Function Approximation: A Gradient
Boosting Machine. \emph{{The Annals of Statistics}}, 29(5), 1189-1232.
\item Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting
System. \emph{{Proceedings of KDD}}.
\item Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting
Decision Tree. \emph{{Advances in Neural Information Processing Systems}}.
\end{{enumerate}}
}}

\end{{document}}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tex, encoding="utf-8")


def compile_latex_report(tex_path: Path) -> Path:
    """Compile the generated LaTeX source and return the PDF path."""

    compiler_environment = os.environ.copy()
    compiler_environment.update({"LANG": "C", "LC_ALL": "C", "LC_CTYPE": "C"})
    try:
        subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ],
            cwd=tex_path.parent,
            env=compiler_environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        subprocess.run(
            ["latexmk", "-c", tex_path.name],
            cwd=tex_path.parent,
            env=compiler_environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("latexmk is required to create the PDF report") from exc
    except subprocess.CalledProcessError as exc:
        compiler_output = exc.stdout or ""
        raise RuntimeError(
            "LaTeX compilation failed:\n" + compiler_output[-6_000:]
        ) from exc
    return tex_path.with_suffix(".pdf")
