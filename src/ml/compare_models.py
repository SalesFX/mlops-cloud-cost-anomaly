#!/usr/bin/env python3
"""Model comparison report generator for the FinOps anomaly detection platform.

Consolidates the outputs of all four detectors into a side-by-side comparison:
  - Statistical Baseline  (rule-based)
  - Isolation Forest      (unsupervised ML)
  - Decision Tree         (supervised ML)
  - XGBoost               (supervised ML)

Usage:
    python src/ml/compare_models.py \\
        --baseline       data/cloud_cost_baseline_predictions.csv \\
        --iforest        data/cloud_cost_iforest_predictions.csv \\
        --decision-tree  data/cloud_cost_decision_tree_predictions.csv \\
        --xgboost        data/cloud_cost_xgboost_predictions.csv \\
        --output-csv     reports/model_comparison.csv \\
        --output-md      reports/model_comparison.md

evaluation_scope = full_dataset_outputs:
    All metrics in this report are computed on the full prediction output of
    each model (10,800 records). Decision Tree and XGBoost had their original
    metrics computed on held-out test splits in Phases 1.5 and 1.6. This report
    consolidates a unified operational view across the complete dataset.
"""

import argparse
from pathlib import Path

import pandas as pd

try:
    from src.ml.data_loading import load_billing_csv  # pytest / installed package
    from src.ml.evaluation import evaluate_binary_classifier
except ImportError:
    from data_loading import load_billing_csv  # python src/ml/compare_models.py
    from evaluation import evaluate_binary_classifier

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_CONFIGS = [
    {
        "name": "Statistical Baseline",
        "type": "rule_based",
        "pred_col": "baseline_anomaly",
        "score_col": "baseline_score",
        "notes": (
            "Rule-based detector using cost/usage thresholds and tag governance flags. "
            "Evaluated on full dataset. Thresholds designed for this synthetic schema."
        ),
    },
    {
        "name": "Isolation Forest",
        "type": "unsupervised_ml",
        "pred_col": "iforest_anomaly",
        "score_col": "iforest_score",
        "notes": (
            "Unsupervised ML — no labels used in training. "
            "Evaluated on full dataset including its own training data."
        ),
    },
    {
        "name": "Decision Tree",
        "type": "supervised_ml",
        "pred_col": "dt_anomaly",
        "score_col": "dt_score",
        "notes": (
            "Supervised ML, explainable. "
            "Original test-split metrics computed in Phase 1.5 (80/20 split). "
            "This report evaluates full dataset outputs."
        ),
    },
    {
        "name": "XGBoost",
        "type": "supervised_ml",
        "pred_col": "xgb_anomaly",
        "score_col": "xgb_score",
        "notes": (
            "Supervised ML, high-performance. "
            "Original test-split metrics computed in Phase 1.6 (80/20 split). "
            "This report evaluates full dataset outputs."
        ),
    },
]

# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_model_metrics(
    df: pd.DataFrame,
    pred_col: str,
    score_col: str,
    model_name: str,
    model_type: str,
    notes: str,
) -> dict:
    """Compute a full metrics dict for one model against is_anomaly ground truth.

    Uses evaluate_binary_classifier from evaluation.py.
    evaluation_scope is always "full_dataset_outputs" in this report.
    """
    y_true = df["is_anomaly"].astype(int).to_numpy()
    y_pred = df[pred_col].astype(bool).astype(int).to_numpy()
    y_score = df[score_col].to_numpy(dtype=float)

    metrics = evaluate_binary_classifier(y_true, y_pred, y_score)
    anomaly_count = int(df[pred_col].astype(bool).sum())

    return {
        "model_name": model_name,
        "model_type": model_type,
        "evaluation_scope": "full_dataset_outputs",
        "anomaly_count": anomaly_count,
        "anomaly_rate": round(anomaly_count / len(df), 4),
        "true_positives": metrics["tp"],
        "false_positives": metrics["fp"],
        "false_negatives": metrics["fn"],
        "true_negatives": metrics["tn"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "notes": notes,
    }


def compare_all_models(
    baseline_df: pd.DataFrame,
    iforest_df: pd.DataFrame,
    dt_df: pd.DataFrame,
    xgb_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run compute_model_metrics for all four models and return a comparison DataFrame."""
    dfs = [baseline_df, iforest_df, dt_df, xgb_df]
    rows = [
        compute_model_metrics(
            df=dfs[i],
            pred_col=cfg["pred_col"],
            score_col=cfg["score_col"],
            model_name=cfg["name"],
            model_type=cfg["type"],
            notes=cfg["notes"],
        )
        for i, cfg in enumerate(_MODEL_CONFIGS)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def save_csv(comparison_df: pd.DataFrame, path: str) -> None:
    """Save the comparison DataFrame to CSV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(p, index=False)


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    """Render a subset of DataFrame columns as a Markdown table."""
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(str(df.at[i, c]) for c in cols) + " |"
        for i in df.index
    ]
    return "\n".join([header, sep] + rows)


def save_markdown(comparison_df: pd.DataFrame, path: str) -> None:
    """Generate and save a Markdown comparison report."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    summary_cols = [
        "model_name", "model_type", "evaluation_scope",
        "anomaly_count", "anomaly_rate",
        "precision", "recall", "f1_score", "roc_auc",
    ]
    detail_cols = [
        "model_name", "true_positives", "false_positives",
        "false_negatives", "true_negatives", "accuracy",
    ]

    best_f1_row = comparison_df.loc[comparison_df["f1_score"].idxmax()]
    best_recall_row = comparison_df.loc[comparison_df["recall"].idxmax()]

    content = f"""\
# Model Comparison Report

**Project:** Cloud Cost Anomaly Detection Platform
**Phase:** 1.7 — Model Comparison
**Evaluation scope:** `full_dataset_outputs`

---

## Overview

This report consolidates a side-by-side comparison of all four anomaly detectors
on the full 10,800-record synthetic billing dataset.

**Scope note:** All metrics below are computed on the **complete prediction output**
of each model (`evaluation_scope = full_dataset_outputs`).
Decision Tree and XGBoost had their primary metrics computed on held-out test splits
(80/20) in Phases 1.5 and 1.6. Those per-phase test-split results are the correct
reference for unbiased supervised model evaluation. This report provides an
operational view across the full dataset for comparison purposes.

---

## Summary

{_md_table(comparison_df, summary_cols)}

**Best F1:** {best_f1_row["model_name"]} ({best_f1_row["f1_score"]})
**Best Recall:** {best_recall_row["model_name"]} ({best_recall_row["recall"]})

---

## Detailed Counts

{_md_table(comparison_df, detail_cols)}

---

## Model Interpretation

### Statistical Baseline
Rule-based detector applying fixed thresholds on rolling cost features and FinOps
governance flags (missing tags, cost spikes above 7d/30d averages). No ML involved.
Strong performance on synthetic data because its thresholds were designed for this
specific schema. In production, thresholds require continuous tuning as cost patterns
evolve.

### Isolation Forest
Unsupervised ML — labels were never used during training. Isolates anomalies by
randomly partitioning the feature space. Evaluated here on the full dataset
including its own training data (no holdout). Its performance reflects the inherent
difficulty of unsupervised detection without label guidance.

### Decision Tree
Supervised ML classifier with constrained depth for explainability. Trained on 80%
of the dataset; original test-split metrics from Phase 1.5. Full-dataset evaluation
in this report includes training data, which inflates apparent performance relative
to the Phase 1.5 test-split results.

### XGBoost
Principal supervised ML classifier using gradient-boosted trees. Trained on 80% of
the dataset; original test-split metrics from Phase 1.6. Expected to achieve the
best overall performance. Full-dataset evaluation here includes training data.

---

## Important Notes

1. **Synthetic dataset.** All metrics are computed on synthetic billing data
   generated with programmatic anomaly injection. Real billing data has no
   ground-truth labels and different statistical properties.

2. **evaluation_scope = full_dataset_outputs.** This report evaluates the complete
   output of each model. Supervised models (Decision Tree, XGBoost) were also
   evaluated on held-out test splits in their respective phases — those results
   are the correct reference for unbiased model selection.

3. **Controlled benchmark only.** These metrics are not a guarantee of production
   performance. Real deployment requires real billing data, continuous validation,
   and monitoring for concept drift.

4. **Baseline advantage in synthetic setting.** The Statistical Baseline was
   calibrated for this synthetic schema. In production, its fixed thresholds would
   need tuning as cost distributions change over time.
"""

    p.write_text(content)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a model comparison report from prediction CSVs."
    )
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--iforest", type=str, required=True)
    parser.add_argument("--decision-tree", type=str, required=True)
    parser.add_argument("--xgboost", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="reports/model_comparison.csv")
    parser.add_argument("--output-md", type=str, default="reports/model_comparison.md")
    args = parser.parse_args()

    baseline_df = load_billing_csv(args.baseline)
    iforest_df = load_billing_csv(args.iforest)
    dt_df = load_billing_csv(args.decision_tree)
    xgb_df = load_billing_csv(args.xgboost)

    comparison = compare_all_models(baseline_df, iforest_df, dt_df, xgb_df)

    save_csv(comparison, args.output_csv)
    save_markdown(comparison, args.output_md)

    print(comparison[["model_name", "precision", "recall", "f1_score", "roc_auc"]].to_string(index=False))
    print(f"\nCSV saved : {args.output_csv}")
    print(f"MD saved  : {args.output_md}")


if __name__ == "__main__":
    main()
