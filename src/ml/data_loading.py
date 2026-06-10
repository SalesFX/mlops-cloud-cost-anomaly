"""Canonical CSV loader for all billing datasets in this project.

Every module in src/ml/ that reads a CSV file should call load_billing_csv()
to ensure consistent NaN handling and type coercion across the pipeline.

Why a shared loader:
    Using keep_default_na=False globally preserves empty tag strings but
    converts numeric NaN (e.g. cost_change_percent on first-day records) to
    empty strings, silently changing column dtypes from float64 to object.
    This loader reads with standard NaN handling and only patches the two
    tag columns that need the empty-string treatment.
"""

import os

import pandas as pd


def load_billing_csv(path: str | os.PathLike) -> pd.DataFrame:
    """Load a billing CSV with correct NaN handling and type coercion.

    Behaviour:
    - Numeric columns retain float NaN (not converted to empty strings).
    - tag_project and tag_owner: empty fields are restored to "" after
      CSV round-trip (pandas converts "" to NaN on write/read).
    - is_anomaly: coerced from "True"/"False" strings back to bool.
    - date: parsed as datetime64.

    Args:
        path: Path to the CSV file. Accepts str, pathlib.Path, or any
              os.PathLike object.

    Returns:
        DataFrame with correct column dtypes.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    for col in ("tag_project", "tag_owner"):
        if col in df.columns:
            df[col] = df[col].fillna("")
    if "is_anomaly" in df.columns:
        df["is_anomaly"] = df["is_anomaly"].astype(str).str.lower() == "true"
    return df
