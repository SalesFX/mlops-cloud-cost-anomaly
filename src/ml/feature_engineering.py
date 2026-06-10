#!/usr/bin/env python3
"""Feature engineering pipeline for the FinOps anomaly detection platform.

Transforms raw billing records into a feature-enriched dataset ready for
statistical baseline and ML model training.

Usage:
    python src/ml/feature_engineering.py \
        --input data/cloud_costs.csv \
        --output data/cloud_cost_features.csv

Grouping note (architecture):
    Rolling features (previous_day_cost, previous_day_usage, avg_cost_7d,
    avg_cost_30d, cost_change_percent, usage_change_percent) are calculated
    per resource_id — the stable identifier generated once per resource in
    the synthetic dataset.

    In real billing data (Phase 6), account_id and region are also stable
    per resource. The recommended group will be:
        provider + account_id + region + resource_id
    This change is isolated to add_rolling_features() and requires no other
    modifications.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.ml.data_loading import load_billing_csv  # pytest / installed package
except ImportError:
    from data_loading import load_billing_csv  # python src/ml/feature_engineering.py

# ---------------------------------------------------------------------------
# Schema contracts
# ---------------------------------------------------------------------------

REQUIRED_INPUT_COLUMNS = [
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

FEATURE_COLUMNS = [
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
# I/O
# ---------------------------------------------------------------------------


def load_dataset(path: str | os.PathLike) -> pd.DataFrame:
    """Load raw billing CSV. Delegates to data_loading.load_billing_csv."""
    return load_billing_csv(path)


def save_features(df: pd.DataFrame, output_path: str) -> None:
    """Save the enriched DataFrame to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError listing any missing required columns."""
    missing = sorted(set(REQUIRED_INPUT_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Feature transformations
# ---------------------------------------------------------------------------


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day_of_week (0=Mon..6=Sun) and is_weekend from the date column."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"] >= 5
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-resource rolling cost and usage features.

    Groups by resource_id (stable synthetic identifier). See module docstring
    for the Phase 6 grouping strategy when real billing data is available.

    Fallback for records without sufficient history: NaN (not zero).
    Rolling windows use min_periods=1 so the first record always has a value.
    """
    df = df.copy()
    # Sort chronologically within each resource group before rolling
    df = df.sort_values(["resource_id", "date"]).reset_index(drop=True)

    grouped_cost = df.groupby("resource_id")["daily_cost"]
    grouped_usage = df.groupby("resource_id")["usage_quantity"]

    df["previous_day_cost"] = grouped_cost.shift(1)
    df["previous_day_usage"] = grouped_usage.shift(1)

    df["avg_cost_7d"] = grouped_cost.transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df["avg_cost_30d"] = grouped_cost.transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )

    # cost_change_percent: NaN when no previous record or previous cost was 0
    df["cost_change_percent"] = np.where(
        df["previous_day_cost"].isna() | (df["previous_day_cost"] == 0),
        np.nan,
        (df["daily_cost"] - df["previous_day_cost"]) / df["previous_day_cost"] * 100,
    )

    # usage_change_percent: derived from previous_day_USAGE (not previous_day_cost)
    df["usage_change_percent"] = np.where(
        df["previous_day_usage"].isna() | (df["previous_day_usage"] == 0),
        np.nan,
        (df["usage_quantity"] - df["previous_day_usage"]) / df["previous_day_usage"] * 100,
    )

    return df


def add_finops_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add FinOps governance flags.

    is_missing_tag: True when tag_project or tag_owner is absent (empty string or NaN).
    cost_to_usage_ratio: NaN when usage_quantity is zero to avoid division by zero.
    """
    df = df.copy()
    df["is_missing_tag"] = (
        (df["tag_project"].fillna("") == "") | (df["tag_owner"].fillna("") == "")
    )
    safe_usage = df["usage_quantity"].replace(0.0, np.nan)
    df["cost_to_usage_ratio"] = df["daily_cost"] / safe_usage
    return df


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature transformations and return the enriched DataFrame.

    Validates required columns, applies temporal, rolling, and FinOps features,
    then restores the original date-first sort order.
    """
    validate_columns(df)
    df = add_temporal_features(df)
    df = add_rolling_features(df)
    df = add_finops_flags(df)
    # Restore date-first sort order (add_rolling_features sorts by resource_id)
    df = df.sort_values(["date", "provider", "service", "environment"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a feature-enriched billing dataset from raw CSV."
    )
    parser.add_argument(
        "--input", type=str, default="data/cloud_costs.csv", help="Raw billing CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/cloud_cost_features.csv",
        help="Output enriched CSV",
    )
    args = parser.parse_args()

    df_raw = load_dataset(args.input)
    df_features = build_features(df_raw)
    save_features(df_features, args.output)

    new_cols = [c for c in FEATURE_COLUMNS if c in df_features.columns]
    print(f"Input rows    : {len(df_raw)}")
    print(f"Output rows   : {len(df_features)}")
    print(f"New columns   : {new_cols}")
    print(f"Output saved  : {args.output}")


if __name__ == "__main__":
    main()
