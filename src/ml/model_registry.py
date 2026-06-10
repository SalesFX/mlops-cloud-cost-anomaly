#!/usr/bin/env python3
"""Model registry for the FinOps anomaly detection platform.

Trains the final XGBoost model on the full dataset and saves three artefacts:
  - models/best_model.joblib       : serialised model for serving
  - models/model_metadata.json     : provenance, hyperparameters, references
  - models/feature_schema.json     : feature contract for the serving API

Usage:
    python src/ml/model_registry.py \\
        --input            data/cloud_cost_features.csv \\
        --model-output     models/best_model.joblib \\
        --metadata-output  models/model_metadata.json \\
        --schema-output    models/feature_schema.json \\
        --model-version    1.0.0 \\
        --seed             42

Training on full dataset:
    The final model is trained on the complete feature dataset (no holdout).
    The objective is the serving artefact, not evaluation. Official evaluation
    metrics come from earlier phases:
      - Phase 1.5: Decision Tree, 80/20 test-split metrics
      - Phase 1.6: XGBoost, 80/20 test-split metrics
      - Phase 1.7: Consolidated comparison report (evaluation_scope=full_dataset_outputs)

Synthetic dataset note:
    All training data is synthetic. Metrics from any phase are a controlled
    benchmark and do not guarantee production performance.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBClassifier

try:
    from src.ml.data_loading import load_billing_csv
    from src.ml.preprocessing import MODEL_FEATURES, build_feature_matrix, build_target
except ImportError:
    from data_loading import load_billing_csv
    from preprocessing import MODEL_FEATURES, build_feature_matrix, build_target

# ---------------------------------------------------------------------------
# Feature schema constants
# ---------------------------------------------------------------------------

BOOLEAN_FEATURES: list[str] = ["is_missing_tag", "is_weekend"]

NUMERIC_FEATURES: list[str] = [
    "daily_cost",
    "usage_quantity",
    "previous_day_cost",
    "previous_day_usage",
    "avg_cost_7d",
    "avg_cost_30d",
    "cost_change_percent",
    "usage_change_percent",
    "cost_to_usage_ratio",
    "day_of_week",
]

_RAW_INPUT_COLUMNS: list[str] = [
    "date",
    "provider",
    "account_id",
    "service",
    "region",
    "environment",
    "resource_id",
    "tag_project",
    "tag_owner",
    "daily_cost",
    "usage_quantity",
    "currency",
    "is_anomaly",
    "anomaly_type",
]

# Default hyperparameters matching Phase 1.6 configuration
_DEFAULT_N_ESTIMATORS = 100
_DEFAULT_MAX_DEPTH = 3
_DEFAULT_LEARNING_RATE = 0.1

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Return n_negatives / n_positives for XGBoost class imbalance handling."""
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    return n_neg / n_pos if n_pos > 0 else 1.0


def train_final_model(
    df,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    seed: int,
) -> XGBClassifier:
    """Train XGBoost on the full dataset for serving.

    is_anomaly is used only as the target y — never as a feature.
    Trains on the complete dataset (no holdout split) because the objective
    is a production artefact, not a held-out evaluation.
    """
    X = build_feature_matrix(df)
    y = build_target(df)
    spw = compute_scale_pos_weight(y)

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
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Metadata and schema builders
# ---------------------------------------------------------------------------


def build_metadata(
    model: XGBClassifier,
    model_version: str,
    training_dataset: str,
    trained_at: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    scale_pos_weight: float,
) -> dict:
    """Build the model_metadata.json content dict.

    trained_at is accepted as a parameter so tests can inject a fixed timestamp.
    The model argument is kept for forward compatibility (future: feature importances).
    """
    return {
        "model_name": "best_model",
        "model_version": model_version,
        "model_type": "supervised_ml",
        "algorithm": "XGBoost",
        "selected_from_report": "reports/model_comparison.csv",
        "training_dataset": training_dataset,
        "trained_at": trained_at,
        "features": MODEL_FEATURES,
        "target": "is_anomaly",
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "scale_pos_weight": round(float(scale_pos_weight), 4),
        },
        "metrics_reference": "reports/model_comparison.csv",
        "evaluation_scope": "final_model_trained_on_full_dataset_for_serving",
        "comparison_report_scope": "Phase 1.7 uses evaluation_scope=full_dataset_outputs",
        "test_split_metrics_reference": {
            "decision_tree": "Phase 1.5",
            "xgboost": "Phase 1.6",
        },
        "notes": (
            "Final model trained on the full synthetic feature dataset for serving. "
            "Official comparison report is available in reports/model_comparison.csv "
            "and reports/model_comparison.md. Production use would require real billing "
            "data, continuous validation, drift monitoring and retraining strategy."
        ),
    }


def build_feature_schema() -> dict:
    """Build the feature_schema.json content dict.

    Separates raw billing columns from engineered model features so the
    future serving API knows exactly what the model expects.
    """
    return {
        "model_features": MODEL_FEATURES,
        "feature_count": len(MODEL_FEATURES),
        "raw_input_columns": _RAW_INPUT_COLUMNS,
        "required_model_features": MODEL_FEATURES,
        "boolean_features": BOOLEAN_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target_column": "is_anomaly",
        "excluded_columns": ["is_anomaly", "anomaly_type"],
        "serving_note": (
            "The current serving contract expects engineered model features. "
            "Future API versions may accept raw billing records and run feature "
            "engineering before prediction."
        ),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(model: XGBClassifier, path: str | os.PathLike) -> None:
    """Serialise the model to disk with joblib."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)


def load_model(path: str | os.PathLike) -> XGBClassifier:
    """Load a joblib-serialised model from disk."""
    return joblib.load(path)


def save_json(data: dict, path: str | os.PathLike) -> None:
    """Write a dict to a JSON file with 2-space indentation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the final XGBoost model and save artefacts for serving."
    )
    parser.add_argument("--input", type=str, default="data/cloud_cost_features.csv")
    parser.add_argument("--model-output", type=str, default="models/best_model.joblib")
    parser.add_argument("--metadata-output", type=str, default="models/model_metadata.json")
    parser.add_argument("--schema-output", type=str, default="models/feature_schema.json")
    parser.add_argument("--model-version", type=str, default="1.0.0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_billing_csv(args.input)
    y = build_target(df)
    spw = compute_scale_pos_weight(y)

    model = train_final_model(
        df,
        n_estimators=_DEFAULT_N_ESTIMATORS,
        max_depth=_DEFAULT_MAX_DEPTH,
        learning_rate=_DEFAULT_LEARNING_RATE,
        seed=args.seed,
    )

    trained_at = datetime.now().isoformat()
    metadata = build_metadata(
        model=model,
        model_version=args.model_version,
        training_dataset=args.input,
        trained_at=trained_at,
        n_estimators=_DEFAULT_N_ESTIMATORS,
        max_depth=_DEFAULT_MAX_DEPTH,
        learning_rate=_DEFAULT_LEARNING_RATE,
        scale_pos_weight=spw,
    )
    schema = build_feature_schema()

    save_model(model, args.model_output)
    save_json(metadata, args.metadata_output)
    save_json(schema, args.schema_output)

    print(f"Model saved    : {args.model_output}")
    print(f"Metadata saved : {args.metadata_output}")
    print(f"Schema saved   : {args.schema_output}")
    print(f"Trained at     : {trained_at}")
    print(f"Features       : {len(MODEL_FEATURES)} ({MODEL_FEATURES})")


if __name__ == "__main__":
    main()
