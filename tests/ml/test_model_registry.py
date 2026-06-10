import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.ml.model_registry import (
    BOOLEAN_FEATURES,
    NUMERIC_FEATURES,
    build_feature_schema,
    build_metadata,
    compute_scale_pos_weight,
    load_model,
    save_json,
    save_model,
    train_final_model,
)
from src.ml.preprocessing import MODEL_FEATURES, build_feature_matrix, build_target

# ---------------------------------------------------------------------------
# Fixture: 100 records, 10 anomalies
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
    return train_final_model(sample_df, n_estimators=10, max_depth=3,
                             learning_rate=0.1, seed=42)


@pytest.fixture
def sample_metadata(sample_df, trained_model):
    y = build_target(sample_df)
    spw = compute_scale_pos_weight(y)
    return build_metadata(
        model=trained_model,
        model_version="1.0.0",
        training_dataset="test.csv",
        trained_at="2026-01-01T00:00:00",
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=spw,
    )


# ---------------------------------------------------------------------------
# train_final_model
# ---------------------------------------------------------------------------

class TestTrainFinalModel:
    def test_returns_xgbclassifier(self, trained_model):
        assert isinstance(trained_model, XGBClassifier)

    def test_model_can_predict_proba(self, sample_df, trained_model):
        X = build_feature_matrix(sample_df)
        proba = trained_model.predict_proba(X)
        assert proba.shape == (len(sample_df), 2)

    def test_predict_proba_sums_to_one(self, sample_df, trained_model):
        X = build_feature_matrix(sample_df)
        proba = trained_model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_is_anomaly_not_used_as_feature(self):
        assert "is_anomaly" not in MODEL_FEATURES

    def test_anomaly_type_not_used_as_feature(self):
        assert "anomaly_type" not in MODEL_FEATURES

    def test_same_seed_same_predictions(self, sample_df):
        m1 = train_final_model(sample_df, 10, 3, 0.1, seed=42)
        m2 = train_final_model(sample_df, 10, 3, 0.1, seed=42)
        X = build_feature_matrix(sample_df)
        np.testing.assert_array_equal(m1.predict(X), m2.predict(X))


# ---------------------------------------------------------------------------
# compute_scale_pos_weight
# ---------------------------------------------------------------------------

class TestComputeScalePosWeight:
    def test_imbalanced_classes(self):
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 9 neg, 1 pos
        assert compute_scale_pos_weight(y) == pytest.approx(9.0)

    def test_balanced_classes(self):
        y = np.array([0, 0, 1, 1])
        assert compute_scale_pos_weight(y) == pytest.approx(1.0)

    def test_no_positives_returns_one(self):
        y = np.array([0, 0, 0])
        assert compute_scale_pos_weight(y) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_metadata
# ---------------------------------------------------------------------------

class TestBuildMetadata:
    def test_required_keys_present(self, sample_metadata):
        for key in ("model_name", "model_version", "model_type", "algorithm",
                    "selected_from_report", "training_dataset", "trained_at",
                    "features", "target", "hyperparameters", "metrics_reference",
                    "evaluation_scope", "comparison_report_scope",
                    "test_split_metrics_reference", "notes"):
            assert key in sample_metadata, f"Missing key: {key}"

    def test_features_equal_model_features(self, sample_metadata):
        assert sample_metadata["features"] == MODEL_FEATURES

    def test_target_is_is_anomaly(self, sample_metadata):
        assert sample_metadata["target"] == "is_anomaly"

    def test_evaluation_scope_correct(self, sample_metadata):
        assert sample_metadata["evaluation_scope"] == "final_model_trained_on_full_dataset_for_serving"

    def test_comparison_report_scope_mentions_full_dataset_outputs(self, sample_metadata):
        assert "full_dataset_outputs" in sample_metadata["comparison_report_scope"]

    def test_test_split_references_phases(self, sample_metadata):
        refs = sample_metadata["test_split_metrics_reference"]
        assert refs["decision_tree"] == "Phase 1.5"
        assert refs["xgboost"] == "Phase 1.6"

    def test_trained_at_injectable(self, sample_df, trained_model):
        y = build_target(sample_df)
        meta = build_metadata(trained_model, "1.0.0", "test.csv",
                              "2026-01-01T00:00:00", 10, 3, 0.1,
                              compute_scale_pos_weight(y))
        assert meta["trained_at"] == "2026-01-01T00:00:00"

    def test_hyperparameters_keys_present(self, sample_metadata):
        for k in ("n_estimators", "max_depth", "learning_rate", "scale_pos_weight"):
            assert k in sample_metadata["hyperparameters"]

    def test_notes_mentions_synthetic(self, sample_metadata):
        assert "synthetic" in sample_metadata["notes"].lower()

    def test_algorithm_is_xgboost(self, sample_metadata):
        assert sample_metadata["algorithm"] == "XGBoost"


# ---------------------------------------------------------------------------
# build_feature_schema
# ---------------------------------------------------------------------------

class TestBuildFeatureSchema:
    @pytest.fixture
    def schema(self):
        return build_feature_schema()

    def test_model_features_present(self, schema):
        assert schema["model_features"] == MODEL_FEATURES

    def test_required_model_features_match(self, schema):
        assert schema["required_model_features"] == MODEL_FEATURES

    def test_feature_count_is_12(self, schema):
        assert schema["feature_count"] == 12

    def test_raw_input_columns_present(self, schema):
        assert "raw_input_columns" in schema
        assert len(schema["raw_input_columns"]) > 0

    def test_boolean_features_correct(self, schema):
        assert set(schema["boolean_features"]) == set(BOOLEAN_FEATURES)

    def test_numeric_features_correct(self, schema):
        assert set(schema["numeric_features"]) == set(NUMERIC_FEATURES)

    def test_boolean_and_numeric_cover_all_features(self, schema):
        all_covered = set(schema["boolean_features"]) | set(schema["numeric_features"])
        assert all_covered == set(MODEL_FEATURES)

    def test_target_column_is_is_anomaly(self, schema):
        assert schema["target_column"] == "is_anomaly"

    def test_excluded_columns_contains_is_anomaly(self, schema):
        assert "is_anomaly" in schema["excluded_columns"]

    def test_excluded_columns_contains_anomaly_type(self, schema):
        assert "anomaly_type" in schema["excluded_columns"]

    def test_serving_note_present(self, schema):
        assert "serving_note" in schema
        assert len(schema["serving_note"]) > 0


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

class TestSaveLoadModel:
    def test_save_creates_file(self, tmp_path, trained_model):
        path = str(tmp_path / "model.joblib")
        save_model(trained_model, path)
        assert Path(path).exists()

    def test_load_returns_xgbclassifier(self, tmp_path, trained_model):
        path = str(tmp_path / "model.joblib")
        save_model(trained_model, path)
        loaded = load_model(path)
        assert isinstance(loaded, XGBClassifier)

    def test_loaded_predictions_match_original(self, tmp_path, sample_df, trained_model):
        X = build_feature_matrix(sample_df)
        path = str(tmp_path / "model.joblib")
        save_model(trained_model, path)
        loaded = load_model(path)
        np.testing.assert_array_equal(trained_model.predict(X), loaded.predict(X))

    def test_save_creates_parent_dirs(self, tmp_path, trained_model):
        path = str(tmp_path / "nested" / "dir" / "model.joblib")
        save_model(trained_model, path)
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# save_json
# ---------------------------------------------------------------------------

class TestSaveJson:
    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "test.json")
        save_json({"key": "value"}, path)
        assert Path(path).exists()

    def test_valid_json_round_trip(self, tmp_path):
        data = {"model": "xgboost", "version": "1.0.0", "count": 42}
        path = str(tmp_path / "test.json")
        save_json(data, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "models" / "sub" / "data.json")
        save_json({"x": 1}, path)
        assert Path(path).exists()

    def test_list_values_preserved(self, tmp_path):
        data = {"features": ["a", "b", "c"]}
        path = str(tmp_path / "test.json")
        save_json(data, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["features"] == ["a", "b", "c"]
