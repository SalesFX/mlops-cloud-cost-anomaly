#!/usr/bin/env python3
"""XGBoost supervised classifier for the FinOps anomaly detection platform.

Principal supervised model in the pipeline. Trained on labeled synthetic data
with is_anomaly as the binary target. Expected to achieve the highest F1 and
ROC-AUC among all supervised models.

Usage:
    python src/ml/xgboost_classifier.py \
        --input  data/cloud_cost_features.csv \
        --output data/cloud_cost_xgboost_predictions.csv \
        --test-size 0.2 \
        --n-estimators 100 \
        --max-depth 3 \
        --learning-rate 0.1 \
        --seed 42

is_anomaly usage:
    is_anomaly is used ONLY as the training target y. It is never included
    in MODEL_FEATURES and is preserved unchanged in the output for evaluation.

Class imbalance:
    scale_pos_weight = n_negatives / n_positives (computed from the training
    split) adjusts the loss function so the minority anomaly class receives
    proportionally higher weight during tree construction.

Evaluation note:
    Metrics (accuracy, precision, recall, F1, ROC-AUC) are calculated on
    the held-out test split only — never on the full dataset or training data.
    Results are a benchmark on a controlled synthetic dataset, not a proxy
    for production performance on real billing data.

Model persistence:
    No model artefact is saved (.pkl / .joblib). Deferred to the model
    registry phase.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    from src.ml.data_loading import load_billing_csv  # pytest / installed package
    from src.ml.preprocessing import MODEL_FEATURES, build_feature_matrix, build_target  # noqa: F401
    from src.ml.evaluation import evaluate_binary_classifier
except ImportError:
    from data_loading import load_billing_csv  # python src/ml/xgboost_classifier.py
    from preprocessing import MODEL_FEATURES, build_feature_matrix, build_target  # noqa: F401
    from evaluation import evaluate_binary_classifier

# Re-exported for test imports that address this module directly.

XGB_OUTPUT_COLUMNS: list[str] = [
    "xgb_anomaly",
    "xgb_score",
    "xgb_risk_level",
]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    seed: int,
) -> XGBClassifier:
    """Fit an XGBClassifier and return the trained model.

    scale_pos_weight = n_neg / n_pos balances the binary cross-entropy loss
    for the ~5% anomaly rate present in both synthetic and real billing data.
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
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
    model: XGBClassifier,
    X: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Run predictions on the full feature matrix and add xgb_* columns.

    Predictions are generated for ALL records, including those used for
    training. Evaluation is isolated to the held-out test set.

    Preserves is_anomaly and anomaly_type unchanged.
    """
    df = df.copy()
    scores = model.predict_proba(X)[:, 1]
    df["xgb_anomaly"] = scores >= 0.5
    df["xgb_score"] = scores
    df["xgb_risk_level"] = [_risk_level(float(s)) for s in scores]
    return df


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def run(
    df: pd.DataFrame,
    test_size: float,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Train, predict on full dataset, evaluate on test split.

    Returns:
        df_out:  Full DataFrame with xgb_* columns for all records.
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

    model = train(X[idx_train], y[idx_train], n_estimators, max_depth, learning_rate, seed)
    df_out = predict_all(model, X, df)

    metrics = evaluate_binary_classifier(
        y_true=y[idx_test],
        y_pred=df_out.iloc[idx_test]["xgb_anomaly"].to_numpy(),
        y_score=df_out.iloc[idx_test]["xgb_score"].to_numpy(),
    )
    return df_out, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost classifier on feature-enriched billing data."
    )
    parser.add_argument("--input", type=str, default="data/cloud_cost_features.csv")
    parser.add_argument("--output", type=str, default="data/cloud_cost_xgboost_predictions.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df_raw = load_billing_csv(args.input)
    df_out, metrics = run(
        df_raw,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)

    n_anomaly = int(df_out["xgb_anomaly"].sum())
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
