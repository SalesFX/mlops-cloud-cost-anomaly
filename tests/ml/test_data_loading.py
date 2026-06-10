import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.data_loading import load_billing_csv


# ---------------------------------------------------------------------------
# Fixture: CSV with numeric NaN and empty tag strings
# ---------------------------------------------------------------------------
#
# Row 0: tag_project="project-alpha", tag_owner="alice",
#        cost_change_percent=NaN (first-day, no previous record), is_anomaly=False
# Row 1: tag_project="",              tag_owner="",
#        cost_change_percent=100.0,   is_anomaly=True (cost_spike)

@pytest.fixture
def billing_csv(tmp_path: Path) -> str:
    df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "tag_project": ["project-alpha", ""],
        "tag_owner": ["alice", ""],
        "daily_cost": [100.0, 200.0],
        "cost_change_percent": [float("nan"), 100.0],
        "is_anomaly": [False, True],
        "anomaly_type": ["none", "cost_spike"],
    })
    path = tmp_path / "billing.csv"
    df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Tag handling
# ---------------------------------------------------------------------------

class TestTagHandling:
    def test_empty_tag_project_becomes_empty_string(self, billing_csv):
        df = load_billing_csv(billing_csv)
        assert df.loc[1, "tag_project"] == ""

    def test_empty_tag_owner_becomes_empty_string(self, billing_csv):
        df = load_billing_csv(billing_csv)
        assert df.loc[1, "tag_owner"] == ""

    def test_empty_tags_are_not_nan(self, billing_csv):
        df = load_billing_csv(billing_csv)
        assert not pd.isna(df.loc[1, "tag_project"])
        assert not pd.isna(df.loc[1, "tag_owner"])

    def test_non_empty_tags_preserved(self, billing_csv):
        df = load_billing_csv(billing_csv)
        assert df.loc[0, "tag_project"] == "project-alpha"
        assert df.loc[0, "tag_owner"] == "alice"

    def test_columns_without_tags_load_fine(self, tmp_path):
        df_no_tags = pd.DataFrame({"date": ["2025-01-01"], "daily_cost": [100.0],
                                   "is_anomaly": [False], "anomaly_type": ["none"]})
        path = tmp_path / "no_tags.csv"
        df_no_tags.to_csv(path, index=False)
        df = load_billing_csv(str(path))
        assert "tag_project" not in df.columns


# ---------------------------------------------------------------------------
# Numeric NaN preservation
# ---------------------------------------------------------------------------

class TestNumericNaN:
    def test_numeric_nan_remains_nan(self, billing_csv):
        df = load_billing_csv(billing_csv)
        assert pd.isna(df.loc[0, "cost_change_percent"])

    def test_numeric_nan_is_not_empty_string(self, billing_csv):
        df = load_billing_csv(billing_csv)
        assert df.loc[0, "cost_change_percent"] != ""

    def test_numeric_column_dtype_is_float(self, billing_csv):
        df = load_billing_csv(billing_csv)
        assert df["cost_change_percent"].dtype == float

    def test_non_nan_numeric_value_preserved(self, billing_csv):
        df = load_billing_csv(billing_csv)
        assert df.loc[1, "cost_change_percent"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------

class TestTypeCoercion:
    def test_is_anomaly_dtype_is_bool(self, billing_csv):
        df = load_billing_csv(billing_csv)
        assert df["is_anomaly"].dtype == bool

    def test_is_anomaly_false_value(self, billing_csv):
        df = load_billing_csv(billing_csv)
        assert df.loc[0, "is_anomaly"] == False

    def test_is_anomaly_true_value(self, billing_csv):
        df = load_billing_csv(billing_csv)
        assert df.loc[1, "is_anomaly"] == True

    def test_date_parsed_as_datetime(self, billing_csv):
        df = load_billing_csv(billing_csv)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_csv_without_is_anomaly_loads_fine(self, tmp_path):
        df_no_flag = pd.DataFrame({"date": ["2025-01-01"], "daily_cost": [100.0]})
        path = tmp_path / "no_flag.csv"
        df_no_flag.to_csv(path, index=False)
        df = load_billing_csv(str(path))
        assert "is_anomaly" not in df.columns


# ---------------------------------------------------------------------------
# Path type compatibility
# ---------------------------------------------------------------------------

class TestPathTypes:
    def test_accepts_str_path(self, billing_csv):
        assert isinstance(billing_csv, str)
        df = load_billing_csv(billing_csv)
        assert len(df) == 2

    def test_accepts_pathlib_path(self, billing_csv):
        df = load_billing_csv(Path(billing_csv))
        assert len(df) == 2

    def test_accepts_os_fspath(self, billing_csv):
        df = load_billing_csv(os.fspath(billing_csv))
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Consistency: load_dataset and load_features must match load_billing_csv
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_load_dataset_matches_load_billing_csv(self, billing_csv):
        from src.ml.feature_engineering import load_dataset
        df_canonical = load_billing_csv(billing_csv)
        df_wrapper = load_dataset(billing_csv)
        pd.testing.assert_frame_equal(df_canonical, df_wrapper)

    def test_load_features_matches_load_billing_csv(self, billing_csv):
        from src.ml.baseline_detector import load_features
        df_canonical = load_billing_csv(billing_csv)
        df_wrapper = load_features(billing_csv)
        pd.testing.assert_frame_equal(df_canonical, df_wrapper)
