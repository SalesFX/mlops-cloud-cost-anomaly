#!/usr/bin/env python3
"""Decision Tree supervised classifier for the FinOps anomaly detection platform.

First supervised model in the pipeline. Trained on labeled synthetic data
with is_anomaly as the binary target. Provides explainable predictions via
feature importance and a constrained tree depth.

Usage:
    python src/ml/decision_tree_classifier.py \
        --input  data/cloud_cost_features.csv \
        --output data/cloud_cost_decision_tree_predictions.csv \
        --test-size  0.2 \
        --max-depth  5 \
        --min-samples-leaf 10 \
        --seed 42

is_anomaly usage:
    is_anomaly is used ONLY as the training target y. It is never included
    in MODEL_FEATURES and is preserved unchanged in the output for evaluation.

Evaluation note:
    Metrics (accuracy, precision, recall, F1, ROC-AUC) are calculated on
    the held-out test split only — never on the full dataset or training data.
    Results are a benchmark on a controlled synthetic dataset, not a proxy
    for production performance on real billing data.

Model persistence:
    No model artefact is saved (.pkl / .joblib). Deferred to the model
    registry phase.

build_feature_matrix note:
    MODEL_FEATURES and build_feature_matrix are intentionally duplicated from
    isolation_forest_detector.py. With two detectors a shared abstraction is
    premature (YAGNI). Extraction to preprocessing.py is planned for Phase 1.6
    when XGBoost is added and the duplication becomes a genuine maintenance risk.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    from src.ml.data_loading import load_billing_csv  # pytest / installed package
    from src.ml.preprocessing import MODEL_FEATURES, build_feature_matrix, build_target  # noqa: F401
    from src.ml.evaluation import evaluate_binary_classifier
except ImportError:
    from data_loading import load_billing_csv  # python src/ml/decision_tree_classifier.py
    from preprocessing import MODEL_FEATURES, build_feature_matrix, build_target  # noqa: F401
    from evaluation import evaluate_binary_classifier

# Re-exported for backward compatibility with existing test imports.
# Tests that do `from src.ml.decision_tree_classifier import MODEL_FEATURES,
# build_feature_matrix, build_target, evaluate_test` continue to work.

DT_OUTPUT_COLUMNS: list[str] = [
    "dt_anomaly",
    "dt_score",
    "dt_risk_level",
]

# Backward-compatible alias — existing tests import evaluate_test from here.
evaluate_test = evaluate_binary_classifier


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int,
    min_samples_leaf: int,
    seed: int,
) -> DecisionTreeClassifier:
    """Fit a Decision Tree classifier and return the trained model.

    class_weight="balanced" adjusts for the class imbalance present in
    real and synthetic billing data (~5% anomaly rate).
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Prediction (full dataset)
# ---------------------------------------------------------------------------


def _risk_level(score: float) -> str:
    if score < 0.4:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def predict_all(
    model: DecisionTreeClassifier,
    X: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Run predictions on the full feature matrix and add dt_* columns.

    Predictions are generated for ALL records, including those used for
    training. This gives a complete output CSV while keeping evaluation
    isolated to the held-out test set.

    Preserves is_anomaly and anomaly_type unchanged.
    """
    df = df.copy()
    proba = model.predict_proba(X)

    # Guard against edge case where only one class appears in training data
    if proba.shape[1] < 2:
        scores = np.zeros(len(X))
    else:
        # Column order follows model.classes_; find the index of class 1 (anomaly)
        anomaly_class_idx = list(model.classes_).index(1) if 1 in model.classes_ else 1
        scores = proba[:, anomaly_class_idx]

    df["dt_anomaly"] = scores >= 0.5
    df["dt_score"] = scores
    df["dt_risk_level"] = [_risk_level(float(s)) for s in scores]
    return df


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def run(
    df: pd.DataFrame,
    test_size: float,
    max_depth: int,
    min_samples_leaf: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Train, predict on full dataset, evaluate on test split.

    Returns:
        df_out:  Full DataFrame with dt_* columns for all records.
        metrics: Evaluation dict calculated on the held-out test split only.
    """
    X = build_feature_matrix(df)
    y = build_target(df)

    indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    model = train(X[idx_train], y[idx_train], max_depth, min_samples_leaf, seed)
    df_out = predict_all(model, X, df)

    metrics = evaluate_test(
        y_true=y[idx_test],
        y_pred=df_out.iloc[idx_test]["dt_anomaly"].to_numpy(),
        y_score=df_out.iloc[idx_test]["dt_score"].to_numpy(),
    )
    return df_out, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Decision Tree classifier on feature-enriched billing data."
    )
    parser.add_argument("--input", type=str, default="data/cloud_cost_features.csv")
    parser.add_argument("--output", type=str, default="data/cloud_cost_decision_tree_predictions.csv")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data for the test split (default: 0.2)")
    parser.add_argument("--max-depth", type=int, default=5,
                        help="Maximum tree depth (default: 5)")
    parser.add_argument("--min-samples-leaf", type=int, default=10,
                        help="Minimum samples per leaf (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    df_raw = load_billing_csv(args.input)
    df_out, metrics = run(
        df_raw,
        test_size=args.test_size,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        seed=args.seed,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)

    n_anomaly = int(df_out["dt_anomaly"].sum())
    print(f"Rows          : {len(df_out)}")
    print(f"Anomalies     : {n_anomaly} ({n_anomaly / len(df_out):.2%})")
    print(f"--- Test set metrics (test_size={args.test_size}) ---")
    print(f"Accuracy      : {metrics['accuracy']:.4f}")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"F1            : {metrics['f1']:.4f}")
    print(f"ROC-AUC       : {metrics['roc_auc']:.4f}")
    print(f"Output        : {args.output}")


if __name__ == "__main__":
    main()
