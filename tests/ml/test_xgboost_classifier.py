import numpy as np
import pandas as pd
import pytest

from src.ml.xgboost_classifier import (
    XGB_OUTPUT_COLUMNS,
    predict_all,
    run,
    train,
)
from src.ml.preprocessing import MODEL_FEATURES, build_feature_matrix, build_target

# ---------------------------------------------------------------------------
# Fixture: 100 records, 10 anomalies (10%) — safe for stratified 80/20 split
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n, n_anom = 100, 10
    dow = list(np.tile(np.arange(7), 15)[:n])
    return pd.DataFrame({
        "daily_cost": rng.lognormal(4.0, 0.5, n),
        "usage_quantity": rng.lognormal(6.0, 0.4, n),
        "previous_day_cost": np.where(np.arange(n) % 60 == 0, np.nan, rng.lognormal(4.0, 0.5, n)),
        "previous_day_usage": np.where(np.arange(n) % 60 == 0, np.nan, rng.lognormal(6.0, 0.4, n)),
        "avg_cost_7d": rng.lognormal(4.0, 0.3, n),
        "avg_cost_30d": rng.lognormal(4.0, 0.2, n),
        "cost_change_percent": np.where(np.arange(n) % 60 == 0, np.nan, rng.normal(5.0, 20.0, n)),
        "usage_change_percent": np.where(np.arange(n) % 60 == 0, np.nan, rng.normal(5.0, 20.0, n)),
        "cost_to_usage_ratio": rng.uniform(0.05, 0.5, n),
        "is_missing_tag": [False] * (n - n_anom) + [True] * n_anom,
        "day_of_week": dow,
        "is_weekend": [d >= 5 for d in dow],
        "is_anomaly": [False] * (n - n_anom) + [True] * n_anom,
        "anomaly_type": ["none"] * (n - n_anom) + ["cost_spike"] * n_anom,
    })


@pytest.fixture
def trained_model(sample_df):
    X = build_feature_matrix(sample_df)
    y = build_target(sample_df)
    return train(X, y, n_estimators=10, max_depth=3, learning_rate=0.1, seed=42)


@pytest.fixture
def predicted_df(sample_df, trained_model):
    X = build_feature_matrix(sample_df)
    return predict_all(trained_model, X, sample_df)


# ---------------------------------------------------------------------------
# Model features contract
# ---------------------------------------------------------------------------

class TestModelFeatures:
    def test_is_anomaly_not_in_model_features(self):
        assert "is_anomaly" not in MODEL_FEATURES

    def test_anomaly_type_not_in_model_features(self):
        assert "anomaly_type" not in MODEL_FEATURES

    def test_feature_count_is_12(self):
        assert len(MODEL_FEATURES) == 12


# ---------------------------------------------------------------------------
# Predict all records
# ---------------------------------------------------------------------------

class TestPredictAll:
    def test_all_output_columns_added(self, predicted_df):
        for col in XGB_OUTPUT_COLUMNS:
            assert col in predicted_df.columns

    def test_no_rows_lost(self, sample_df, predicted_df):
        assert len(predicted_df) == len(sample_df)

    def test_xgb_anomaly_is_bool(self, predicted_df):
        assert predicted_df["xgb_anomaly"].dtype == bool

    def test_xgb_score_in_range_0_to_1(self, predicted_df):
        assert (predicted_df["xgb_score"] >= 0.0).all()
        assert (predicted_df["xgb_score"] <= 1.0).all()

    def test_xgb_risk_level_valid_values(self, predicted_df):
        assert set(predicted_df["xgb_risk_level"].unique()).issubset({"low", "medium", "high"})

    def test_is_anomaly_preserved(self, sample_df, predicted_df):
        pd.testing.assert_series_equal(
            predicted_df["is_anomaly"].reset_index(drop=True),
            sample_df["is_anomaly"].reset_index(drop=True),
        )

    def test_anomaly_type_preserved(self, sample_df, predicted_df):
        pd.testing.assert_series_equal(
            predicted_df["anomaly_type"].reset_index(drop=True),
            sample_df["anomaly_type"].reset_index(drop=True),
        )

    def test_input_not_mutated(self, sample_df, trained_model):
        X = build_feature_matrix(sample_df)
        original_cols = list(sample_df.columns)
        predict_all(trained_model, X, sample_df)
        assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Full pipeline (run)
# ---------------------------------------------------------------------------

class TestRun:
    def test_returns_tuple(self, sample_df):
        result = run(sample_df, test_size=0.2, n_estimators=10, max_depth=3,
                     learning_rate=0.1, seed=42)
        assert isinstance(result, tuple) and len(result) == 2

    def test_df_has_output_columns(self, sample_df):
        df_out, _ = run(sample_df, test_size=0.2, n_estimators=10, max_depth=3,
                        learning_rate=0.1, seed=42)
        for col in XGB_OUTPUT_COLUMNS:
            assert col in df_out.columns

    def test_no_rows_lost(self, sample_df):
        df_out, _ = run(sample_df, test_size=0.2, n_estimators=10, max_depth=3,
                        learning_rate=0.1, seed=42)
        assert len(df_out) == len(sample_df)

    def test_is_anomaly_preserved(self, sample_df):
        df_out, _ = run(sample_df, test_size=0.2, n_estimators=10, max_depth=3,
                        learning_rate=0.1, seed=42)
        pd.testing.assert_series_equal(
            df_out["is_anomaly"].reset_index(drop=True),
            sample_df["is_anomaly"].reset_index(drop=True),
        )

    def test_metrics_dict_has_required_keys(self, sample_df):
        _, metrics = run(sample_df, test_size=0.2, n_estimators=10, max_depth=3,
                         learning_rate=0.1, seed=42)
        for k in ("tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1", "roc_auc"):
            assert k in metrics


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_labels(self, sample_df):
        df1, _ = run(sample_df, test_size=0.2, n_estimators=10, max_depth=3,
                     learning_rate=0.1, seed=42)
        df2, _ = run(sample_df, test_size=0.2, n_estimators=10, max_depth=3,
                     learning_rate=0.1, seed=42)
        pd.testing.assert_series_equal(df1["xgb_anomaly"], df2["xgb_anomaly"])

    def test_same_seed_same_scores(self, sample_df):
        df1, _ = run(sample_df, test_size=0.2, n_estimators=10, max_depth=3,
                     learning_rate=0.1, seed=42)
        df2, _ = run(sample_df, test_size=0.2, n_estimators=10, max_depth=3,
                     learning_rate=0.1, seed=42)
        pd.testing.assert_series_equal(df1["xgb_score"], df2["xgb_score"])
