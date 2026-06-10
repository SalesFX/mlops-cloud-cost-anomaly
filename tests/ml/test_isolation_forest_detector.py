import numpy as np
import pandas as pd
import pytest

from src.ml.isolation_forest_detector import (
    IFOREST_OUTPUT_COLUMNS,
    MODEL_FEATURES,
    build_feature_matrix,
    evaluate,
    predict,
    run,
    score_to_normalized,
    train,
)

# ---------------------------------------------------------------------------
# Fixture: 20 records with NaN in first-day rolling features and 2 anomalies
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 20
    day_of_week = list(np.tile(np.arange(7), 3)[:n])
    return pd.DataFrame({
        "daily_cost": rng.lognormal(4.0, 0.5, n),
        "usage_quantity": rng.lognormal(6.0, 0.4, n),
        "previous_day_cost": np.where(
            np.arange(n) == 0, np.nan, rng.lognormal(4.0, 0.5, n)
        ),
        "previous_day_usage": np.where(
            np.arange(n) == 0, np.nan, rng.lognormal(6.0, 0.4, n)
        ),
        "avg_cost_7d": rng.lognormal(4.0, 0.3, n),
        "avg_cost_30d": rng.lognormal(4.0, 0.2, n),
        "cost_change_percent": np.where(
            np.arange(n) == 0, np.nan, rng.normal(5.0, 20.0, n)
        ),
        "usage_change_percent": np.where(
            np.arange(n) == 0, np.nan, rng.normal(5.0, 20.0, n)
        ),
        "cost_to_usage_ratio": rng.uniform(0.05, 0.5, n),
        "is_missing_tag": [False] * 18 + [True, True],
        "day_of_week": day_of_week,
        "is_weekend": [d >= 5 for d in day_of_week],
        "is_anomaly": [False] * 18 + [True, True],
        "anomaly_type": ["none"] * 18 + ["cost_spike", "missing_tag"],
    })


# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------

class TestModelFeatures:
    def test_is_anomaly_not_in_model_features(self):
        assert "is_anomaly" not in MODEL_FEATURES

    def test_anomaly_type_not_in_model_features(self):
        assert "anomaly_type" not in MODEL_FEATURES

    def test_model_features_count(self):
        assert len(MODEL_FEATURES) == 12

    def test_expected_features_present(self):
        for col in ("daily_cost", "usage_quantity", "avg_cost_7d", "avg_cost_30d",
                    "cost_change_percent", "usage_change_percent", "cost_to_usage_ratio",
                    "is_missing_tag", "day_of_week", "is_weekend"):
            assert col in MODEL_FEATURES


# ---------------------------------------------------------------------------
# Feature matrix preparation
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_output_shape(self, sample_df):
        X = build_feature_matrix(sample_df)
        assert X.shape == (len(sample_df), len(MODEL_FEATURES))

    def test_no_nan_in_output(self, sample_df):
        X = build_feature_matrix(sample_df)
        assert not np.isnan(X).any()

    def test_output_is_float64(self, sample_df):
        X = build_feature_matrix(sample_df)
        assert X.dtype == np.float64

    def test_bool_columns_cast_to_numeric(self, sample_df):
        X = build_feature_matrix(sample_df)
        idx_missing = MODEL_FEATURES.index("is_missing_tag")
        unique_vals = set(X[:, idx_missing])
        assert unique_vals.issubset({0.0, 1.0})

    def test_is_anomaly_absent_from_matrix(self, sample_df):
        # Matrix column count must match MODEL_FEATURES only
        X = build_feature_matrix(sample_df)
        assert X.shape[1] == len(MODEL_FEATURES)

    def test_nan_filled_not_propagated(self, sample_df):
        # Row 0 has NaN in previous_day_cost, previous_day_usage, etc.
        X = build_feature_matrix(sample_df)
        assert not np.isnan(X[0]).any()

    def test_input_dataframe_not_mutated(self, sample_df):
        original_cols = list(sample_df.columns)
        build_feature_matrix(sample_df)
        assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------

class TestScoreToNormalized:
    def test_minimum_is_zero(self):
        raw = np.array([-0.5, -0.3, -0.2, -0.1])
        assert score_to_normalized(raw).min() == pytest.approx(0.0)

    def test_maximum_is_one(self):
        raw = np.array([-0.5, -0.3, -0.2, -0.1])
        assert score_to_normalized(raw).max() == pytest.approx(1.0)

    def test_most_anomalous_raw_gets_score_one(self):
        # Most negative raw → highest normalised score
        raw = np.array([-0.5, -0.3, -0.1])
        assert score_to_normalized(raw)[0] == pytest.approx(1.0)

    def test_least_anomalous_raw_gets_score_zero(self):
        raw = np.array([-0.5, -0.3, -0.1])
        assert score_to_normalized(raw)[2] == pytest.approx(0.0)

    def test_order_preserved_inversely(self):
        raw = np.array([-0.5, -0.3, -0.1])
        n = score_to_normalized(raw)
        assert n[0] > n[1] > n[2]

    def test_constant_scores_return_zeros(self):
        raw = np.array([-0.3, -0.3, -0.3])
        assert (score_to_normalized(raw) == 0.0).all()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TestPredict:
    def test_all_output_columns_added(self, sample_df):
        X = build_feature_matrix(sample_df)
        model = train(X, contamination=0.1, seed=42)
        df = predict(model, X, sample_df)
        for col in IFOREST_OUTPUT_COLUMNS:
            assert col in df.columns

    def test_no_rows_lost(self, sample_df):
        X = build_feature_matrix(sample_df)
        model = train(X, contamination=0.1, seed=42)
        df = predict(model, X, sample_df)
        assert len(df) == len(sample_df)

    def test_iforest_anomaly_is_bool(self, sample_df):
        X = build_feature_matrix(sample_df)
        model = train(X, contamination=0.1, seed=42)
        df = predict(model, X, sample_df)
        assert df["iforest_anomaly"].dtype == bool

    def test_iforest_score_in_range(self, sample_df):
        X = build_feature_matrix(sample_df)
        model = train(X, contamination=0.1, seed=42)
        df = predict(model, X, sample_df)
        assert (df["iforest_score"] >= 0.0).all()
        assert (df["iforest_score"] <= 1.0).all()

    def test_iforest_risk_level_valid_values(self, sample_df):
        X = build_feature_matrix(sample_df)
        model = train(X, contamination=0.1, seed=42)
        df = predict(model, X, sample_df)
        assert set(df["iforest_risk_level"].unique()).issubset({"low", "medium", "high"})

    def test_is_anomaly_preserved(self, sample_df):
        X = build_feature_matrix(sample_df)
        model = train(X, contamination=0.1, seed=42)
        df = predict(model, X, sample_df)
        pd.testing.assert_series_equal(
            df["is_anomaly"].reset_index(drop=True),
            sample_df["is_anomaly"].reset_index(drop=True),
        )

    def test_anomaly_type_preserved(self, sample_df):
        X = build_feature_matrix(sample_df)
        model = train(X, contamination=0.1, seed=42)
        df = predict(model, X, sample_df)
        pd.testing.assert_series_equal(
            df["anomaly_type"].reset_index(drop=True),
            sample_df["anomaly_type"].reset_index(drop=True),
        )

    def test_input_dataframe_not_mutated(self, sample_df):
        X = build_feature_matrix(sample_df)
        model = train(X, contamination=0.1, seed=42)
        original_cols = list(sample_df.columns)
        predict(model, X, sample_df)
        assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_required_keys(self):
        df = pd.DataFrame({
            "iforest_anomaly": [True, True, False, False],
            "is_anomaly": [True, False, True, False],
        })
        result = evaluate(df)
        for key in ("tp", "fp", "fn", "tn", "precision", "recall", "f1"):
            assert key in result

    def test_correct_counts(self):
        df = pd.DataFrame({
            "iforest_anomaly": [True, True, False, False],
            "is_anomaly": [True, False, True, False],
        })
        result = evaluate(df)
        assert result["tp"] == 1
        assert result["fp"] == 1
        assert result["fn"] == 1
        assert result["tn"] == 1

    def test_precision_recall_f1_equal_prediction(self):
        df = pd.DataFrame({
            "iforest_anomaly": [True, True, False, False],
            "is_anomaly": [True, False, True, False],
        })
        result = evaluate(df)
        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == pytest.approx(0.5)
        assert result["f1"] == pytest.approx(0.5)

    def test_perfect_precision(self):
        df = pd.DataFrame({
            "iforest_anomaly": [True, False, False, False],
            "is_anomaly": [True, True, False, False],
        })
        result = evaluate(df)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(0.5)

    def test_zero_division_safe_no_positives_predicted(self):
        df = pd.DataFrame({
            "iforest_anomaly": [False, False],
            "is_anomaly": [True, False],
        })
        result = evaluate(df)
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"] == pytest.approx(0.0)
        assert result["f1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_identical_anomaly_labels(self, sample_df):
        X = build_feature_matrix(sample_df)
        df1 = predict(train(X, 0.1, 42), X, sample_df)
        df2 = predict(train(X, 0.1, 42), X, sample_df)
        pd.testing.assert_series_equal(df1["iforest_anomaly"], df2["iforest_anomaly"])

    def test_same_seed_identical_scores(self, sample_df):
        X = build_feature_matrix(sample_df)
        df1 = predict(train(X, 0.1, 42), X, sample_df)
        df2 = predict(train(X, 0.1, 42), X, sample_df)
        pd.testing.assert_series_equal(df1["iforest_score"], df2["iforest_score"])


# ---------------------------------------------------------------------------
# Full pipeline (run)
# ---------------------------------------------------------------------------

class TestRun:
    def test_adds_all_output_columns(self, sample_df):
        df = run(sample_df, contamination=0.1, seed=42)
        for col in IFOREST_OUTPUT_COLUMNS:
            assert col in df.columns

    def test_no_rows_lost(self, sample_df):
        df = run(sample_df, contamination=0.1, seed=42)
        assert len(df) == len(sample_df)

    def test_is_anomaly_preserved_end_to_end(self, sample_df):
        df = run(sample_df, contamination=0.1, seed=42)
        pd.testing.assert_series_equal(
            df["is_anomaly"].reset_index(drop=True),
            sample_df["is_anomaly"].reset_index(drop=True),
        )
