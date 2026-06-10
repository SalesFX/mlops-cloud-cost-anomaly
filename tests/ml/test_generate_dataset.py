import pytest
import pandas as pd

from src.ml.generate_dataset import (
    ANOMALY_TYPES_LIST,
    ENVIRONMENTS,
    PROVIDERS,
    REQUIRED_COLUMNS,
    RESOURCES_PER_COMBO,
    generate_dataset,
)

FIXED_END = "2025-12-31"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestRequiredColumns:
    def test_all_required_columns_present(self):
        df = generate_dataset(days=7, seed=42, anomaly_rate=0.05, end_date=FIXED_END)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_exact_column_set(self):
        df = generate_dataset(days=7, seed=42, anomaly_rate=0.05, end_date=FIXED_END)
        assert set(df.columns) == set(REQUIRED_COLUMNS)


# ---------------------------------------------------------------------------
# Anomaly label consistency
# ---------------------------------------------------------------------------

class TestAnomalyConsistency:
    def test_anomalous_records_have_non_none_type(self):
        df = generate_dataset(days=30, seed=42, anomaly_rate=0.05, end_date=FIXED_END)
        assert (df.loc[df["is_anomaly"], "anomaly_type"] != "none").all()

    def test_normal_records_have_none_type(self):
        df = generate_dataset(days=30, seed=42, anomaly_rate=0.05, end_date=FIXED_END)
        assert (df.loc[~df["is_anomaly"], "anomaly_type"] == "none").all()

    def test_missing_tag_has_empty_tags(self):
        df = generate_dataset(days=90, seed=42, anomaly_rate=0.10, end_date=FIXED_END)
        missing = df[df["anomaly_type"] == "missing_tag"]
        if not missing.empty:
            assert (missing["tag_project"] == "").all()
            assert (missing["tag_owner"] == "").all()

    def test_cost_spike_is_higher_than_normal_max(self):
        normal_df = generate_dataset(days=30, seed=42, anomaly_rate=0.0, end_date=FIXED_END)
        spike_df = generate_dataset(days=30, seed=42, anomaly_rate=0.20, end_date=FIXED_END)
        spikes = spike_df[spike_df["anomaly_type"] == "cost_spike"]
        if not spikes.empty:
            assert spikes["daily_cost"].max() > normal_df["daily_cost"].max()

    def test_anomaly_types_are_valid_values(self):
        df = generate_dataset(days=30, seed=42, anomaly_rate=0.10, end_date=FIXED_END)
        valid = set(ANOMALY_TYPES_LIST) | {"none"}
        assert set(df["anomaly_type"].unique()).issubset(valid)


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------

class TestDataIntegrity:
    def test_daily_cost_positive(self):
        df = generate_dataset(days=7, seed=42, anomaly_rate=0.05, end_date=FIXED_END)
        assert (df["daily_cost"] > 0).all()

    def test_usage_quantity_positive(self):
        df = generate_dataset(days=7, seed=42, anomaly_rate=0.05, end_date=FIXED_END)
        assert (df["usage_quantity"] > 0).all()

    def test_currency_is_usd(self):
        df = generate_dataset(days=7, seed=42, anomaly_rate=0.05, end_date=FIXED_END)
        assert (df["currency"] == "USD").all()

    def test_providers_are_valid(self):
        df = generate_dataset(days=7, seed=42, anomaly_rate=0.05, end_date=FIXED_END)
        assert set(df["provider"].unique()).issubset(set(PROVIDERS.keys()))

    def test_all_providers_present(self):
        df = generate_dataset(days=7, seed=42, anomaly_rate=0.05, end_date=FIXED_END)
        assert set(df["provider"].unique()) == set(PROVIDERS.keys())

    def test_environments_are_valid(self):
        df = generate_dataset(days=7, seed=42, anomaly_rate=0.05, end_date=FIXED_END)
        assert set(df["environment"].unique()).issubset(set(ENVIRONMENTS))

    def test_all_environments_present(self):
        df = generate_dataset(days=7, seed=42, anomaly_rate=0.05, end_date=FIXED_END)
        assert set(df["environment"].unique()) == set(ENVIRONMENTS)

    def test_no_null_values_in_required_columns(self):
        df = generate_dataset(days=7, seed=42, anomaly_rate=0.05, end_date=FIXED_END)
        # tag columns may be empty string for missing_tag anomalies but never NaN
        assert df[REQUIRED_COLUMNS].isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# Record count
# ---------------------------------------------------------------------------

class TestRecordCount:
    def test_record_count_matches_formula(self):
        days = 7
        total_services = sum(len(p["services"]) for p in PROVIDERS.values())
        expected = days * total_services * len(ENVIRONMENTS) * RESOURCES_PER_COMBO
        df = generate_dataset(days=days, seed=42, anomaly_rate=0.05, end_date=FIXED_END)
        assert len(df) == expected

    def test_doubling_days_doubles_records(self):
        df7 = generate_dataset(days=7, seed=42, anomaly_rate=0.0, end_date=FIXED_END)
        df14 = generate_dataset(days=14, seed=42, anomaly_rate=0.0, end_date=FIXED_END)
        assert len(df14) == 2 * len(df7)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_produces_identical_dataframes(self):
        df1 = generate_dataset(days=7, seed=42, anomaly_rate=0.05, end_date=FIXED_END)
        df2 = generate_dataset(days=7, seed=42, anomaly_rate=0.05, end_date=FIXED_END)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_costs(self):
        df1 = generate_dataset(days=7, seed=42, anomaly_rate=0.05, end_date=FIXED_END)
        df2 = generate_dataset(days=7, seed=99, anomaly_rate=0.05, end_date=FIXED_END)
        assert not df1["daily_cost"].equals(df2["daily_cost"])


# ---------------------------------------------------------------------------
# Anomaly rate
# ---------------------------------------------------------------------------

class TestAnomalyRate:
    def test_zero_rate_produces_no_anomalies(self):
        df = generate_dataset(days=30, seed=42, anomaly_rate=0.0, end_date=FIXED_END)
        assert df["is_anomaly"].sum() == 0

    def test_anomaly_rate_within_tolerance(self):
        df = generate_dataset(days=60, seed=42, anomaly_rate=0.05, end_date=FIXED_END)
        actual = df["is_anomaly"].mean()
        assert 0.03 <= actual <= 0.07, f"Anomaly rate {actual:.3f} outside expected range"

    def test_invalid_anomaly_rate_raises(self):
        with pytest.raises(ValueError):
            generate_dataset(days=7, seed=42, anomaly_rate=1.5, end_date=FIXED_END)


# ---------------------------------------------------------------------------
# Environment cost ordering
# ---------------------------------------------------------------------------

class TestEnvironmentCosts:
    def test_prod_costs_exceed_dev(self):
        df = generate_dataset(days=30, seed=42, anomaly_rate=0.0, end_date=FIXED_END)
        prod_mean = df[df["environment"] == "prod"]["daily_cost"].mean()
        dev_mean = df[df["environment"] == "dev"]["daily_cost"].mean()
        assert prod_mean > dev_mean

    def test_staging_costs_between_dev_and_prod(self):
        df = generate_dataset(days=30, seed=42, anomaly_rate=0.0, end_date=FIXED_END)
        prod_mean = df[df["environment"] == "prod"]["daily_cost"].mean()
        staging_mean = df[df["environment"] == "staging"]["daily_cost"].mean()
        dev_mean = df[df["environment"] == "dev"]["daily_cost"].mean()
        assert dev_mean < staging_mean < prod_mean
