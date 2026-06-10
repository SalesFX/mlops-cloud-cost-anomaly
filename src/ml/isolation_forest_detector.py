#!/usr/bin/env python3
"""Isolation Forest anomaly detector for the FinOps anomaly detection platform.

Unsupervised anomaly detection using IsolationForest from scikit-learn.
First ML model in the pipeline — trained without labels.

Usage:
    python src/ml/isolation_forest_detector.py \
        --input  data/cloud_cost_features.csv \
        --output data/cloud_cost_iforest_predictions.csv \
        --contamination 0.05 \
        --seed 42

Note on is_missing_tag as a feature:
    is_missing_tag is included in MODEL_FEATURES as an explicit FinOps governance
    signal. Records with empty cost-allocation tags form a distinct cluster in
    feature space, making missing_tag anomalies easier to isolate without labels.
    In production, missing tags are a known bad practice and worth flagging
    unsupervised.

Note on evaluation:
    evaluate() compares iforest_anomaly against the is_anomaly ground truth
    present in the synthetic dataset. These metrics are a benchmark for the
    synthetic scenario only — not a proxy for production performance. In real
    billing data (Phase 6), no labels exist and Isolation Forest operates
    fully unsupervised.

Model persistence:
    No model artefact is saved in this phase (.pkl / .joblib).
    Persistence and versioning are deferred to the model registry phase.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

try:
    from src.ml.data_loading import load_billing_csv  # pytest / installed package
    from src.ml.preprocessing import MODEL_FEATURES, build_feature_matrix  # noqa: F401
except ImportError:
    from data_loading import load_billing_csv  # python src/ml/isolation_forest_detector.py
    from preprocessing import MODEL_FEATURES, build_feature_matrix  # noqa: F401

# Re-exported for backward compatibility with existing test imports.
# Tests that do `from src.ml.isolation_forest_detector import MODEL_FEATURES`
# will continue to work because the names are in this module's namespace.

IFOREST_OUTPUT_COLUMNS: list[str] = [
    "iforest_anomaly",
    "iforest_score",
    "iforest_risk_level",
]

# ---------------------------------------------------------------------------
# Feature preparation — delegated to preprocessing.py
# ---------------------------------------------------------------------------
#
# build_feature_matrix is imported above and re-exported.
# The docstring below is preserved for context.
#
# build_feature_matrix(df):
#   1. Selects MODEL_FEATURES — excludes is_anomaly and anomaly_type.
#   2. Casts boolean columns (is_missing_tag, is_weekend) to int.
#   3. Fills NaN with per-column median — covers first-day records that have
#      no rolling history (previous_day_cost, cost_change_percent, etc.).


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    X: np.ndarray,
    contamination: float,
    seed: int,
) -> IsolationForest:
    """Fit an IsolationForest on the feature matrix and return the model."""
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=seed,
    )
    model.fit(X)
    return model


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_to_normalized(raw_scores: np.ndarray) -> np.ndarray:
    """Normalise IsolationForest.score_samples() output to [0.0, 1.0].

    score_samples() convention: more negative = more anomalous.
    After flipping sign and min-max normalisation:
        0.0 = least anomalous record in the dataset
        1.0 = most anomalous record in the dataset
    """
    flipped = -raw_scores
    s_min, s_max = flipped.min(), flipped.max()
    if s_max == s_min:
        return np.zeros_like(flipped, dtype=float)
    return (flipped - s_min) / (s_max - s_min)


def _risk_level(score: float) -> str:
    if score < 0.4:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict(
    model: IsolationForest,
    X: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add iforest_* columns to a copy of df.

    is_anomaly and anomaly_type are carried through unchanged — they are
    ground-truth labels for evaluation, not model inputs.
    """
    df = df.copy()
    labels = model.predict(X)           # -1 = anomaly, 1 = normal
    scores = score_to_normalized(model.score_samples(X))

    df["iforest_anomaly"] = labels == -1
    df["iforest_score"] = scores
    df["iforest_risk_level"] = [_risk_level(float(s)) for s in scores]
    return df


# ---------------------------------------------------------------------------
# Evaluation (synthetic benchmark only)
# ---------------------------------------------------------------------------


def evaluate(df: pd.DataFrame) -> dict:
    """Compare iforest_anomaly to is_anomaly ground truth.

    Valid only on the synthetic dataset where labels exist.
    Not applicable in production (Phase 6).
    """
    actual = df["is_anomaly"].astype(bool)
    predicted = df["iforest_anomaly"].astype(bool)

    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def run(
    df: pd.DataFrame,
    contamination: float,
    seed: int,
) -> pd.DataFrame:
    """Build feature matrix, train, predict. Returns df enriched with iforest_* columns."""
    X = build_feature_matrix(df)
    model = train(X, contamination, seed)
    return predict(model, X, df)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Isolation Forest anomaly detection on feature-enriched billing data."
    )
    parser.add_argument("--input", type=str, default="data/cloud_cost_features.csv")
    parser.add_argument("--output", type=str, default="data/cloud_cost_iforest_predictions.csv")
    parser.add_argument(
        "--contamination", type=float, default=0.05,
        help="Expected fraction of anomalies (default: 0.05)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    df_raw = load_billing_csv(args.input)
    df_out = run(df_raw, contamination=args.contamination, seed=args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)

    metrics = evaluate(df_out)
    n_anomaly = int(df_out["iforest_anomaly"].sum())

    print(f"Rows          : {len(df_out)}")
    print(f"Anomalies     : {n_anomaly} ({n_anomaly / len(df_out):.2%})")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"F1            : {metrics['f1']:.4f}")
    print(f"Output        : {args.output}")


if __name__ == "__main__":
    main()
