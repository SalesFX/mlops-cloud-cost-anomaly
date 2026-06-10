import numpy as np
import pandas as pd
import pytest

from src.ml.feature_engineering import (
    FEATURE_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    add_finops_flags,
    add_rolling_features,
    add_temporal_features,
    build_features,
    load_dataset,
    validate_columns,
)

# ---------------------------------------------------------------------------
# Shared fixture: 2 resources x 5 days with known values
# ---------------------------------------------------------------------------
#
# Resource 1 (AWS EC2 prod, i-aaa00001):
#   costs  = [100.0, 110.0,  90.0, 200.0, 105.0]
#   usage  = [500.0, 550.0, 480.0, 600.0, 510.0]
#   day 3 (2025-01-03): missing tags (empty strings)
#   day 4 (2025-01-04): cost_spike anomaly, is_anomaly=True
#
# Resource 2 (OCI Compute dev, ocid1.instance.oc1.bbb):
#   all normal, tags always present
#
# Date weekdays:
#   2025-01-01 Wed (dayofweek=2)   not weekend
#   2025-01-02 Thu (dayofweek=3)   not weekend
#   2025-01-03 Fri (dayofweek=4)   not weekend
#   2025-01-04 Sat (dayofweek=5)   weekend
#   2025-01-05 Sun (dayofweek=6)   weekend

R1 = "i-aaa00001"
R2 = "ocid1.instance.oc1.bbb"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    records = [
        # (date, provider, account_id, service, region, env, resource_id,
        #  tag_project, tag_owner, daily_cost, usage_quantity,
        #  currency, is_anomaly, anomaly_type)
        ("2025-01-01", "AWS", "123456789012", "EC2", "us-east-1", "prod", R1,
         "project-alpha", "alice", 100.0, 500.0, "USD", False, "none"),
        ("2025-01-02", "AWS", "123456789012", "EC2", "us-east-1", "prod", R1,
         "project-alpha", "alice", 110.0, 550.0, "USD", False, "none"),
        ("2025-01-03", "AWS", "123456789012", "EC2", "us-east-1", "prod", R1,
         "", "", 90.0, 480.0, "USD", False, "none"),   # missing tags
        ("2025-01-04", "AWS", "123456789012", "EC2", "us-east-1", "prod", R1,
         "project-alpha", "alice", 200.0, 600.0, "USD", True, "cost_spike"),
        ("2025-01-05", "AWS", "123456789012", "EC2", "us-east-1", "prod", R1,
         "project-alpha", "alice", 105.0, 510.0, "USD", False, "none"),
        ("2025-01-01", "OCI", "ocid1.tenancy.oc1..aaa123456789", "Compute",
         "us-ashburn-1", "dev", R2, "project-beta", "bob", 20.0, 100.0, "USD", False, "none"),
        ("2025-01-02", "OCI", "ocid1.tenancy.oc1..aaa123456789", "Compute",
         "us-ashburn-1", "dev", R2, "project-beta", "bob", 22.0, 110.0, "USD", False, "none"),
        ("2025-01-03", "OCI", "ocid1.tenancy.oc1..aaa123456789", "Compute",
         "us-ashburn-1", "dev", R2, "project-beta", "bob", 18.0, 95.0, "USD", False, "none"),
        ("2025-01-04", "OCI", "ocid1.tenancy.oc1..aaa123456789", "Compute",
         "us-ashburn-1", "dev", R2, "project-beta", "bob", 21.0, 105.0, "USD", False, "none"),
        ("2025-01-05", "OCI", "ocid1.tenancy.oc1..aaa123456789", "Compute",
         "us-ashburn-1", "dev", R2, "project-beta", "bob", 23.0, 115.0, "USD", False, "none"),
    ]
    columns = REQUIRED_INPUT_COLUMNS
    df = pd.DataFrame(records, columns=columns)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _row(df: pd.DataFrame, date: str, resource_id: str) -> pd.Series:
    mask = (df["date"] == pd.Timestamp(date)) & (df["resource_id"] == resource_id)
    return df[mask].iloc[0]


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------

class TestValidateColumns:
    def test_valid_dataframe_does_not_raise(self, sample_df):
        validate_columns(sample_df)

    def test_missing_column_raises_value_error(self, sample_df):
        with pytest.raises(ValueError, match="daily_cost"):
            validate_columns(sample_df.drop(columns=["daily_cost"]))

    def test_error_message_names_all_missing_columns(self, sample_df):
        bad = sample_df.drop(columns=["daily_cost", "usage_quantity"])
        with pytest.raises(ValueError) as exc_info:
            validate_columns(bad)
        msg = str(exc_info.value)
        assert "daily_cost" in msg
        assert "usage_quantity" in msg


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

class TestLoad:
    def test_empty_tag_strings_preserved_not_converted_to_nan(self, tmp_path, sample_df):
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)
        df = load_dataset(str(csv_path))
        r1_day3 = _row(df, "2025-01-03", R1)
        assert r1_day3["tag_project"] == ""
        assert not pd.isna(r1_day3["tag_project"])

    def test_date_column_parsed_as_datetime(self, tmp_path, sample_df):
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)
        df = load_dataset(str(csv_path))
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_is_anomaly_loaded_as_bool(self, tmp_path, sample_df):
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)
        df = load_dataset(str(csv_path))
        assert df["is_anomaly"].dtype == bool

    def test_normal_tags_still_present_after_load(self, tmp_path, sample_df):
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)
        df = load_dataset(str(csv_path))
        r1_day1 = _row(df, "2025-01-01", R1)
        assert r1_day1["tag_project"] == "project-alpha"


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------

class TestTemporalFeatures:
    def test_wednesday_day_of_week(self, sample_df):
        df = add_temporal_features(sample_df)
        assert _row(df, "2025-01-01", R1)["day_of_week"] == 2

    def test_saturday_day_of_week(self, sample_df):
        df = add_temporal_features(sample_df)
        assert _row(df, "2025-01-04", R1)["day_of_week"] == 5

    def test_sunday_day_of_week(self, sample_df):
        df = add_temporal_features(sample_df)
        assert _row(df, "2025-01-05", R1)["day_of_week"] == 6

    def test_weekday_is_not_weekend(self, sample_df):
        df = add_temporal_features(sample_df)
        assert _row(df, "2025-01-01", R1)["is_weekend"] == False

    def test_saturday_is_weekend(self, sample_df):
        df = add_temporal_features(sample_df)
        assert _row(df, "2025-01-04", R1)["is_weekend"] == True

    def test_sunday_is_weekend(self, sample_df):
        df = add_temporal_features(sample_df)
        assert _row(df, "2025-01-05", R1)["is_weekend"] == True

    def test_input_dataframe_not_mutated(self, sample_df):
        original_cols = list(sample_df.columns)
        add_temporal_features(sample_df)
        assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------

class TestRollingFeatures:
    def test_first_day_previous_day_cost_is_nan(self, sample_df):
        df = add_rolling_features(sample_df)
        assert pd.isna(_row(df, "2025-01-01", R1)["previous_day_cost"])

    def test_first_day_previous_day_usage_is_nan(self, sample_df):
        df = add_rolling_features(sample_df)
        assert pd.isna(_row(df, "2025-01-01", R1)["previous_day_usage"])

    def test_second_day_previous_day_cost(self, sample_df):
        df = add_rolling_features(sample_df)
        assert _row(df, "2025-01-02", R1)["previous_day_cost"] == pytest.approx(100.0)

    def test_second_day_previous_day_usage(self, sample_df):
        df = add_rolling_features(sample_df)
        assert _row(df, "2025-01-02", R1)["previous_day_usage"] == pytest.approx(500.0)

    def test_avg_cost_7d_first_day_equals_cost(self, sample_df):
        df = add_rolling_features(sample_df)
        assert _row(df, "2025-01-01", R1)["avg_cost_7d"] == pytest.approx(100.0)

    def test_avg_cost_7d_second_day(self, sample_df):
        df = add_rolling_features(sample_df)
        # (100 + 110) / 2 = 105.0
        assert _row(df, "2025-01-02", R1)["avg_cost_7d"] == pytest.approx(105.0)

    def test_avg_cost_7d_fifth_day(self, sample_df):
        df = add_rolling_features(sample_df)
        # (100 + 110 + 90 + 200 + 105) / 5 = 121.0
        assert _row(df, "2025-01-05", R1)["avg_cost_7d"] == pytest.approx(121.0)

    def test_avg_cost_30d_equals_7d_when_fewer_than_30_days(self, sample_df):
        df = add_rolling_features(sample_df)
        r = _row(df, "2025-01-05", R1)
        assert r["avg_cost_30d"] == pytest.approx(r["avg_cost_7d"])

    def test_cost_change_percent_first_day_is_nan(self, sample_df):
        df = add_rolling_features(sample_df)
        assert pd.isna(_row(df, "2025-01-01", R1)["cost_change_percent"])

    def test_cost_change_percent_second_day(self, sample_df):
        df = add_rolling_features(sample_df)
        # (110 - 100) / 100 * 100 = 10.0
        assert _row(df, "2025-01-02", R1)["cost_change_percent"] == pytest.approx(10.0)

    def test_usage_change_percent_first_day_is_nan(self, sample_df):
        df = add_rolling_features(sample_df)
        assert pd.isna(_row(df, "2025-01-01", R1)["usage_change_percent"])

    def test_usage_change_percent_second_day_derived_from_previous_usage(self, sample_df):
        df = add_rolling_features(sample_df)
        # (550 - 500) / 500 * 100 = 10.0 — uses previous_day_USAGE, not previous_day_cost
        assert _row(df, "2025-01-02", R1)["usage_change_percent"] == pytest.approx(10.0)

    def test_usage_change_percent_fourth_day(self, sample_df):
        df = add_rolling_features(sample_df)
        # usage day 4 = 600, prev usage = 480 -> (600-480)/480*100 = 25.0
        assert _row(df, "2025-01-04", R1)["usage_change_percent"] == pytest.approx(25.0)

    def test_rolling_features_isolated_between_resources(self, sample_df):
        df = add_rolling_features(sample_df)
        # Resource 2 day 1: previous_day_cost must be NaN (not resource 1's last cost)
        assert pd.isna(_row(df, "2025-01-01", R2)["previous_day_cost"])

    def test_resource_2_rolling_independent_from_resource_1(self, sample_df):
        df = add_rolling_features(sample_df)
        # Resource 2 day 2: previous_day_cost = 20.0 (its own day 1)
        assert _row(df, "2025-01-02", R2)["previous_day_cost"] == pytest.approx(20.0)

    def test_input_dataframe_not_mutated(self, sample_df):
        original_len = len(sample_df)
        add_rolling_features(sample_df)
        assert len(sample_df) == original_len


# ---------------------------------------------------------------------------
# FinOps flags
# ---------------------------------------------------------------------------

class TestFinopsFlags:
    def test_missing_tag_true_when_both_empty(self, sample_df):
        df = add_finops_flags(sample_df)
        assert _row(df, "2025-01-03", R1)["is_missing_tag"] == True

    def test_missing_tag_false_when_both_present(self, sample_df):
        df = add_finops_flags(sample_df)
        assert _row(df, "2025-01-01", R1)["is_missing_tag"] == False

    def test_missing_tag_true_when_only_project_empty(self, sample_df):
        df = sample_df.copy()
        df.loc[(df["resource_id"] == R1) & (df["date"] == pd.Timestamp("2025-01-02")), "tag_project"] = ""
        df = add_finops_flags(df)
        assert _row(df, "2025-01-02", R1)["is_missing_tag"] == True

    def test_missing_tag_true_when_only_owner_empty(self, sample_df):
        df = sample_df.copy()
        df.loc[(df["resource_id"] == R1) & (df["date"] == pd.Timestamp("2025-01-02")), "tag_owner"] = ""
        df = add_finops_flags(df)
        assert _row(df, "2025-01-02", R1)["is_missing_tag"] == True

    def test_cost_to_usage_ratio_value(self, sample_df):
        df = add_finops_flags(sample_df)
        # 100.0 / 500.0 = 0.2
        assert _row(df, "2025-01-01", R1)["cost_to_usage_ratio"] == pytest.approx(0.2)

    def test_cost_to_usage_ratio_nan_when_zero_usage(self, sample_df):
        df = sample_df.copy()
        df.loc[(df["resource_id"] == R1) & (df["date"] == pd.Timestamp("2025-01-01")), "usage_quantity"] = 0.0
        df = add_finops_flags(df)
        assert pd.isna(_row(df, "2025-01-01", R1)["cost_to_usage_ratio"])

    def test_input_dataframe_not_mutated(self, sample_df):
        original_cols = list(sample_df.columns)
        add_finops_flags(sample_df)
        assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Full pipeline: build_features
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_all_feature_columns_present(self, sample_df):
        df = build_features(sample_df)
        for col in FEATURE_COLUMNS:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_no_rows_lost(self, sample_df):
        df = build_features(sample_df)
        assert len(df) == len(sample_df)

    def test_original_columns_preserved(self, sample_df):
        df = build_features(sample_df)
        for col in REQUIRED_INPUT_COLUMNS:
            assert col in df.columns

    def test_is_anomaly_preserved(self, sample_df):
        df = build_features(sample_df)
        assert _row(df, "2025-01-04", R1)["is_anomaly"] == True

    def test_anomaly_type_preserved(self, sample_df):
        df = build_features(sample_df)
        assert _row(df, "2025-01-04", R1)["anomaly_type"] == "cost_spike"

    def test_normal_anomaly_type_preserved(self, sample_df):
        df = build_features(sample_df)
        assert _row(df, "2025-01-01", R1)["anomaly_type"] == "none"
        assert _row(df, "2025-01-01", R1)["is_anomaly"] == False

    def test_missing_column_raises_value_error(self, sample_df):
        with pytest.raises(ValueError):
            build_features(sample_df.drop(columns=["daily_cost"]))

    def test_exactly_ten_new_feature_columns(self, sample_df):
        df = build_features(sample_df)
        assert len(FEATURE_COLUMNS) == 10
        for col in FEATURE_COLUMNS:
            assert col in df.columns
