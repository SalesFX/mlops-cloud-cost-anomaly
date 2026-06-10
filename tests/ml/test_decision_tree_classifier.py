import numpy as np
import pandas as pd
import pytest

from src.ml.decision_tree_classifier import (
    DT_OUTPUT_COLUMNS,
    MODEL_FEATURES,
    build_feature_matrix,
    build_target,
    evaluate_test,
    predict_all,
    run,
    train,
)

# ---------------------------------------------------------------------------
# Fixture: 100 records, 10 anomalies (10%) — safe for stratified 80/20 split
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n, n_anomaly = 100, 10
    day_of_week = list(np.tile(np.arange(7), 15)[:n])
    return pd.DataFrame({
        "daily_cost": rng.lognormal(4.0, 0.5, n),
        "usage_quantity": rng.lognormal(6.0, 0.4, n),
        "previous_day_cost": np.where(
            np.arange(n) % 60 == 0, np.nan, rng.lognormal(4.0, 0.5, n)
        ),
        "previous_day_usage": np.where(
            np.arange(n) % 60 == 0, np.nan, rng.lognormal(6.0, 0.4, n)
        ),
        "avg_cost_7d": rng.lognormal(4.0, 0.3, n),
        "avg_cost_30d": rng.lognormal(4.0, 0.2, n),
        "cost_change_percent": np.where(
            np.arange(n) % 60 == 0, np.nan, rng.normal(5.0, 20.0, n)
        ),
        "usage_change_percent": np.where(
            np.arange(n) % 60 == 0, np.nan, rng.normal(5.0, 20.0, n)
        ),
        "cost_to_usage_ratio": rng.uniform(0.05, 0.5, n),
        "is_missing_tag": [False] * (n - n_anomaly) + [True] * n_anomaly,
        "day_of_week": day_of_week,
        "is_weekend": [d >= 5 for d in day_of_week],
        "is_anomaly": [False] * (n - n_anomaly) + [True] * n_anomaly,
        "anomaly_type": ["none"] * (n - n_anomaly) + ["cost_spike"] * n_anomaly,
    })


@pytest.fixture
def trained_model(sample_df):
    X = build_feature_matrix(sample_df)
    y = build_target(sample_df)
    return train(X, y, max_depth=5, min_samples_leaf=1, seed=42)


@pytest.fixture
def predicted_df(sample_df, trained_model):
    X = build_feature_matrix(sample_df)
    return predict_all(trained_model, X, sample_df)


# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------

class TestModelFeatures:
    def test_is_anomaly_not_in_model_features(self):
        assert "is_anomaly" not in MODEL_FEATURES

    def test_anomaly_type_not_in_model_features(self):
        assert "anomaly_type" not in MODEL_FEATURES

    def test_feature_count_is_12(self):
        assert len(MODEL_FEATURES) == 12


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------

class TestBuildTarget:
    def test_output_is_numpy_array(self, sample_df):
        y = build_target(sample_df)
        assert isinstance(y, np.ndarray)

    def test_values_are_0_and_1_only(self, sample_df):
        y = build_target(sample_df)
        assert set(y).issubset({0, 1})

    def test_positive_count_matches_is_anomaly(self, sample_df):
        y = build_target(sample_df)
        assert y.sum() == sample_df["is_anomaly"].sum()

    def test_length_matches_dataframe(self, sample_df):
        y = build_target(sample_df)
        assert len(y) == len(sample_df)

    def test_input_not_mutated(self, sample_df):
        original_cols = list(sample_df.columns)
        build_target(sample_df)
        assert list(sample_df.columns) == original_cols


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

    def test_bool_columns_cast_to_0_or_1(self, sample_df):
        X = build_feature_matrix(sample_df)
        idx = MODEL_FEATURES.index("is_missing_tag")
        assert set(X[:, idx]).issubset({0.0, 1.0})

    def test_nan_filled_for_first_day_records(self, sample_df):
        X = build_feature_matrix(sample_df)
        assert not np.isnan(X[0]).any()

    def test_input_not_mutated(self, sample_df):
        original_cols = list(sample_df.columns)
        build_feature_matrix(sample_df)
        assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Predict (full dataset)
# ---------------------------------------------------------------------------

class TestPredictAll:
    def test_all_output_columns_added(self, predicted_df):
        for col in DT_OUTPUT_COLUMNS:
            assert col in predicted_df.columns

    def test_no_rows_lost(self, sample_df, predicted_df):
        assert len(predicted_df) == len(sample_df)

    def test_dt_anomaly_is_bool(self, predicted_df):
        assert predicted_df["dt_anomaly"].dtype == bool

    def test_dt_score_in_range_0_to_1(self, predicted_df):
        assert (predicted_df["dt_score"] >= 0.0).all()
        assert (predicted_df["dt_score"] <= 1.0).all()

    def test_dt_risk_level_valid_values(self, predicted_df):
        assert set(predicted_df["dt_risk_level"].unique()).issubset({"low", "medium", "high"})

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
# Evaluation (decoupled signature)
# ---------------------------------------------------------------------------

class TestEvaluateTest:
    def test_returns_required_keys(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.6, 0.3, 0.1])
        result = evaluate_test(y_true, y_pred, y_score)
        for key in ("tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1", "roc_auc"):
            assert key in result

    def test_correct_counts(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.6, 0.3, 0.1])
        r = evaluate_test(y_true, y_pred, y_score)
        assert r["tp"] == 1
        assert r["fp"] == 1
        assert r["fn"] == 1
        assert r["tn"] == 1

    def test_precision_recall_f1(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.6, 0.3, 0.1])
        r = evaluate_test(y_true, y_pred, y_score)
        assert r["precision"] == pytest.approx(0.5)
        assert r["recall"] == pytest.approx(0.5)
        assert r["f1"] == pytest.approx(0.5)

    def test_perfect_predictions(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        y_score = np.array([0.9, 0.1, 0.8, 0.2])
        r = evaluate_test(y_true, y_pred, y_score)
        assert r["accuracy"] == pytest.approx(1.0)
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(1.0)
        assert r["f1"] == pytest.approx(1.0)

    def test_roc_auc_in_range(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.6, 0.3, 0.1])
        r = evaluate_test(y_true, y_pred, y_score)
        assert 0.0 <= r["roc_auc"] <= 1.0

    def test_zero_division_safe_no_positives_predicted(self):
        y_true = np.array([1, 0])
        y_pred = np.array([0, 0])
        y_score = np.array([0.3, 0.2])
        r = evaluate_test(y_true, y_pred, y_score)
        assert r["precision"] == pytest.approx(0.0)
        assert r["f1"] == pytest.approx(0.0)

    def test_roc_auc_graceful_single_class_in_true(self):
        # roc_auc_score raises ValueError when only one class in y_true
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 0, 1])
        y_score = np.array([0.8, 0.3, 0.7])
        r = evaluate_test(y_true, y_pred, y_score)
        assert r["roc_auc"] == pytest.approx(0.0)

    def test_accepts_bool_arrays(self):
        y_true = np.array([True, False, True, False])
        y_pred = np.array([True, False, True, False])
        y_score = np.array([0.9, 0.1, 0.8, 0.2])
        r = evaluate_test(y_true, y_pred, y_score)
        assert r["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Full pipeline (run)
# ---------------------------------------------------------------------------

class TestRun:
    def test_returns_tuple(self, sample_df):
        result = run(sample_df, test_size=0.2, max_depth=5, min_samples_leaf=1, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_df_has_output_columns(self, sample_df):
        df_out, _ = run(sample_df, test_size=0.2, max_depth=5, min_samples_leaf=1, seed=42)
        for col in DT_OUTPUT_COLUMNS:
            assert col in df_out.columns

    def test_no_rows_lost(self, sample_df):
        df_out, _ = run(sample_df, test_size=0.2, max_depth=5, min_samples_leaf=1, seed=42)
        assert len(df_out) == len(sample_df)

    def test_is_anomaly_preserved(self, sample_df):
        df_out, _ = run(sample_df, test_size=0.2, max_depth=5, min_samples_leaf=1, seed=42)
        pd.testing.assert_series_equal(
            df_out["is_anomaly"].reset_index(drop=True),
            sample_df["is_anomaly"].reset_index(drop=True),
        )

    def test_metrics_dict_has_required_keys(self, sample_df):
        _, metrics = run(sample_df, test_size=0.2, max_depth=5, min_samples_leaf=1, seed=42)
        for k in ("tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1", "roc_auc"):
            assert k in metrics


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_anomaly_labels(self, sample_df):
        df1, _ = run(sample_df, test_size=0.2, max_depth=5, min_samples_leaf=1, seed=42)
        df2, _ = run(sample_df, test_size=0.2, max_depth=5, min_samples_leaf=1, seed=42)
        pd.testing.assert_series_equal(df1["dt_anomaly"], df2["dt_anomaly"])

    def test_same_seed_same_scores(self, sample_df):
        df1, _ = run(sample_df, test_size=0.2, max_depth=5, min_samples_leaf=1, seed=42)
        df2, _ = run(sample_df, test_size=0.2, max_depth=5, min_samples_leaf=1, seed=42)
        pd.testing.assert_series_equal(df1["dt_score"], df2["dt_score"])
