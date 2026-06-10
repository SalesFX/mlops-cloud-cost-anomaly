import numpy as np
import pandas as pd
import pytest

from src.ml.preprocessing import (
    MODEL_FEATURES,
    build_feature_matrix,
    build_target,
)

# ---------------------------------------------------------------------------
# Fixture: 50 records, 5 anomalies (10%)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n, n_anom = 50, 5
    dow = list(np.tile(np.arange(7), 8)[:n])
    return pd.DataFrame({
        "daily_cost": rng.lognormal(4.0, 0.5, n),
        "usage_quantity": rng.lognormal(6.0, 0.4, n),
        "previous_day_cost": np.where(np.arange(n) % 50 == 0, np.nan, rng.lognormal(4.0, 0.5, n)),
        "previous_day_usage": np.where(np.arange(n) % 50 == 0, np.nan, rng.lognormal(6.0, 0.4, n)),
        "avg_cost_7d": rng.lognormal(4.0, 0.3, n),
        "avg_cost_30d": rng.lognormal(4.0, 0.2, n),
        "cost_change_percent": np.where(np.arange(n) % 50 == 0, np.nan, rng.normal(5.0, 20.0, n)),
        "usage_change_percent": np.where(np.arange(n) % 50 == 0, np.nan, rng.normal(5.0, 20.0, n)),
        "cost_to_usage_ratio": rng.uniform(0.05, 0.5, n),
        "is_missing_tag": [False] * (n - n_anom) + [True] * n_anom,
        "day_of_week": dow,
        "is_weekend": [d >= 5 for d in dow],
        "is_anomaly": [False] * (n - n_anom) + [True] * n_anom,
        "anomaly_type": ["none"] * (n - n_anom) + ["cost_spike"] * n_anom,
    })


class TestModelFeatures:
    def test_is_anomaly_excluded(self):
        assert "is_anomaly" not in MODEL_FEATURES

    def test_anomaly_type_excluded(self):
        assert "anomaly_type" not in MODEL_FEATURES

    def test_exactly_12_features(self):
        assert len(MODEL_FEATURES) == 12

    def test_required_fields_present(self):
        for col in ("daily_cost", "usage_quantity", "avg_cost_7d", "avg_cost_30d",
                    "is_missing_tag", "day_of_week", "is_weekend"):
            assert col in MODEL_FEATURES


class TestBuildFeatureMatrix:
    def test_shape(self, sample_df):
        X = build_feature_matrix(sample_df)
        assert X.shape == (len(sample_df), len(MODEL_FEATURES))

    def test_no_nan(self, sample_df):
        X = build_feature_matrix(sample_df)
        assert not np.isnan(X).any()

    def test_float64_dtype(self, sample_df):
        X = build_feature_matrix(sample_df)
        assert X.dtype == np.float64

    def test_bool_cast_to_0_or_1(self, sample_df):
        X = build_feature_matrix(sample_df)
        idx = MODEL_FEATURES.index("is_missing_tag")
        assert set(X[:, idx]).issubset({0.0, 1.0})

    def test_nan_filled_for_row_zero(self, sample_df):
        X = build_feature_matrix(sample_df)
        assert not np.isnan(X[0]).any()

    def test_input_not_mutated(self, sample_df):
        original_cols = list(sample_df.columns)
        build_feature_matrix(sample_df)
        assert list(sample_df.columns) == original_cols


class TestBuildTarget:
    def test_output_is_ndarray(self, sample_df):
        y = build_target(sample_df)
        assert isinstance(y, np.ndarray)

    def test_values_are_0_or_1(self, sample_df):
        y = build_target(sample_df)
        assert set(y).issubset({0, 1})

    def test_positive_count_matches(self, sample_df):
        y = build_target(sample_df)
        assert int(y.sum()) == int(sample_df["is_anomaly"].sum())

    def test_length_matches_dataframe(self, sample_df):
        y = build_target(sample_df)
        assert len(y) == len(sample_df)

    def test_input_not_mutated(self, sample_df):
        original_cols = list(sample_df.columns)
        build_target(sample_df)
        assert list(sample_df.columns) == original_cols
