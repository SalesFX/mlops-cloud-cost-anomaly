#!/usr/bin/env python3
"""Statistical baseline anomaly detector for the FinOps anomaly detection platform.

Applies rule-based detection over feature-engineered billing records.
Serves as the non-ML baseline for comparison against Isolation Forest,
Decision Tree and XGBoost in later phases.

Usage:
    python src/ml/baseline_detector.py \
        --input  data/cloud_cost_features.csv \
        --output data/cloud_cost_baseline_predictions.csv

Rule priority (lowest to highest — higher-priority reason overwrites lower):
    missing_tag < high_usage_change < high_cost_change
    < cost_above_7d_average < cost_above_30d_average

cost_above_30d_average has the highest priority because it reflects a violation
of the long-term cost baseline, which is a stronger anomaly signal than a short-
term spike that may still be within the historical norm.
"""

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Schema contracts
# ---------------------------------------------------------------------------

REQUIRED_INPUT_COLUMNS = [
    "daily_cost",
    "avg_cost_7d",
    "avg_cost_30d",
    "cost_change_percent",
    "usage_change_percent",
    "is_missing_tag",
    "is_anomaly",
    "anomaly_type",
]

BASELINE_OUTPUT_COLUMNS = [
    "baseline_anomaly",
    "baseline_score",
    "baseline_risk_level",
    "baseline_reason",
]

# ---------------------------------------------------------------------------
# Thresholds and scoring
# ---------------------------------------------------------------------------

THRESHOLDS: dict[str, float] = {
    "cost_multiplier_7d": 2.5,
    "cost_multiplier_30d": 3.0,
    "cost_change_pct": 150.0,
    "usage_change_pct": 200.0,
}

# Fixed score per reason; also defines the valid reason values
REASON_SCORE: dict[str, float] = {
    "none": 0.0,
    "missing_tag": 0.30,
    "high_usage_change": 0.50,
    "high_cost_change": 0.60,
    "cost_above_7d_average": 0.75,
    "cost_above_30d_average": 0.85,
}


def _risk_level(score: float) -> str:
    if score < 0.3:
        return "low"
    if score < 0.6:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all baseline rules and add four baseline_* columns.

    Rules are applied in priority order using df.loc[] assignments —
    each assignment overwrites the previous for records where the rule fires.
    The last assignment in source order has the highest priority.

    Preserves is_anomaly and anomaly_type unchanged.
    """
    df = df.copy()

    # Evaluate rules; fillna(False) ensures NaN comparisons never raise or
    # accidentally flag first-day records that lack rolling history.
    rule_missing = (df["is_missing_tag"] == True)

    rule_usage_pct = (
        df["usage_change_percent"] >= THRESHOLDS["usage_change_pct"]
    ).fillna(False)

    rule_cost_pct = (
        df["cost_change_percent"] >= THRESHOLDS["cost_change_pct"]
    ).fillna(False)

    rule_7d = (
        df["daily_cost"] > df["avg_cost_7d"] * THRESHOLDS["cost_multiplier_7d"]
    ).fillna(False)

    rule_30d = (
        df["daily_cost"] > df["avg_cost_30d"] * THRESHOLDS["cost_multiplier_30d"]
    ).fillna(False)

    # Apply in ascending priority — later assignments overwrite earlier ones
    df["baseline_reason"] = "none"
    df.loc[rule_missing, "baseline_reason"] = "missing_tag"           # priority 1 (lowest)
    df.loc[rule_usage_pct, "baseline_reason"] = "high_usage_change"   # priority 2
    df.loc[rule_cost_pct, "baseline_reason"] = "high_cost_change"     # priority 3
    df.loc[rule_7d, "baseline_reason"] = "cost_above_7d_average"      # priority 4
    df.loc[rule_30d, "baseline_reason"] = "cost_above_30d_average"    # priority 5 (highest)

    df["baseline_anomaly"] = df["baseline_reason"] != "none"
    df["baseline_score"] = df["baseline_reason"].map(REASON_SCORE)
    df["baseline_risk_level"] = df["baseline_score"].apply(_risk_level)

    return df


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_features(path: str) -> pd.DataFrame:
    """Load feature-enriched billing CSV.

    Reads with standard NaN handling so numeric columns (cost_change_percent,
    usage_change_percent, etc.) retain float NaN for first-day records.
    Tag columns are then patched: CSV round-trip converts empty strings to NaN,
    so we restore them to "" to preserve the is_missing_tag semantics.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    for col in ("tag_project", "tag_owner"):
        if col in df.columns:
            df[col] = df[col].fillna("")
    df["is_anomaly"] = df["is_anomaly"].astype(str).str.lower() == "true"
    return df


def save_predictions(df: pd.DataFrame, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply statistical baseline anomaly detection to feature-enriched billing data."
    )
    parser.add_argument("--input", type=str, default="data/cloud_cost_features.csv")
    parser.add_argument("--output", type=str, default="data/cloud_cost_baseline_predictions.csv")
    args = parser.parse_args()

    df = load_features(args.input)
    df = detect(df)
    save_predictions(df, args.output)

    print(f"Rows          : {len(df)}")
    print(f"Anomalies     : {df['baseline_anomaly'].sum()} ({df['baseline_anomaly'].mean():.2%})")
    print(f"Output        : {args.output}")


if __name__ == "__main__":
    main()
