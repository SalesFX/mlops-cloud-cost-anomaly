"""Shared feature preprocessing for all ML detectors in this project.

Extracted in Phase 1.6 when Isolation Forest, Decision Tree and XGBoost
all needed the same MODEL_FEATURES and build_feature_matrix logic.

All detector modules (isolation_forest_detector, decision_tree_classifier,
xgboost_classifier) import from here and re-export these names so that
existing test imports remain valid.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------

# is_anomaly and anomaly_type are NEVER included here.
MODEL_FEATURES: list[str] = [
    "daily_cost",
    "usage_quantity",
    "previous_day_cost",
    "previous_day_usage",
    "avg_cost_7d",
    "avg_cost_30d",
    "cost_change_percent",
    "usage_change_percent",
    "cost_to_usage_ratio",
    "is_missing_tag",
    "day_of_week",
    "is_weekend",
]

# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract and preprocess MODEL_FEATURES for any ML detector.

    1. Selects MODEL_FEATURES — excludes is_anomaly and anomaly_type.
    2. Casts boolean columns (is_missing_tag, is_weekend) to int.
    3. Fills NaN with per-column median (covers first-day records that lack
       rolling history: previous_day_cost, cost_change_percent, etc.).

    Returns float64 ndarray of shape (n_records, len(MODEL_FEATURES)).
    """
    X = df[MODEL_FEATURES].copy()
    for col in ("is_missing_tag", "is_weekend"):
        if col in X.columns:
            X[col] = X[col].astype(int)
    X = X.fillna(X.median())
    return X.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------


def build_target(df: pd.DataFrame) -> np.ndarray:
    """Extract is_anomaly as a binary int array (0=normal, 1=anomaly).

    is_anomaly is used only as the supervised training target y.
    It is never passed to build_feature_matrix.
    """
    return df["is_anomaly"].astype(int).to_numpy()
