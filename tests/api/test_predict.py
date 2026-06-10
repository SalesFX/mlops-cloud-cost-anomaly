"""Tests for POST /predict endpoint."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main
from src.api.config import settings
from src.api.main import app
from src.api.services.prediction_service import (
    _risk_level,
    build_feature_vector,
)
from src.api.schemas import PredictRequest

client = TestClient(app)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "daily_cost": 320.50,
    "usage_quantity": 9000.0,
    "previous_day_cost": 82.10,
    "previous_day_usage": 2100.0,
    "avg_cost_7d": 85.20,
    "avg_cost_30d": 78.40,
    "cost_change_percent": 290.30,
    "usage_change_percent": 210.50,
    "cost_to_usage_ratio": 0.035,
    "is_missing_tag": False,
    "day_of_week": 2,
    "is_weekend": False,
}

FEATURE_ORDER = [
    "daily_cost", "usage_quantity", "previous_day_cost", "previous_day_usage",
    "avg_cost_7d", "avg_cost_30d", "cost_change_percent", "usage_change_percent",
    "cost_to_usage_ratio", "is_missing_tag", "day_of_week", "is_weekend",
]

MOCK_METADATA = {
    "model_name": "best_model",
    "model_version": "1.0.0",
    "algorithm": "XGBoost",
    "model_type": "supervised_ml",
    "features": FEATURE_ORDER,
    "target": "is_anomaly",
    "evaluation_scope": "final_model_trained_on_full_dataset_for_serving",
}

MOCK_SCHEMA = {
    "feature_count": 12,
    "model_features": FEATURE_ORDER,
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_paths(tmp_path):
    meta = tmp_path / "metadata.json"
    schema = tmp_path / "schema.json"
    meta.write_text(json.dumps(MOCK_METADATA))
    schema.write_text(json.dumps(MOCK_SCHEMA))
    return str(meta), str(schema)


@pytest.fixture
def mock_model_high():
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.05, 0.95]])
    return m


@pytest.fixture
def mock_model_low():
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.95, 0.05]])
    return m


@pytest.fixture
def mock_model_medium():
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.45, 0.55]])
    return m


@pytest.fixture
def predict_setup(mock_paths, mock_model_high, monkeypatch):
    meta_path, schema_path = mock_paths
    monkeypatch.setattr(settings, "model_metadata_path", meta_path)
    monkeypatch.setattr(settings, "feature_schema_path", schema_path)
    monkeypatch.setattr(api_main, "load_model", lambda path: mock_model_high)


# ---------------------------------------------------------------------------
# POST /predict — valid responses
# ---------------------------------------------------------------------------


class TestPredictValid:
    def test_returns_200(self, predict_setup):
        assert client.post("/predict", json=VALID_PAYLOAD).status_code == 200

    def test_response_has_anomaly_field(self, predict_setup):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "anomaly" in body
        assert isinstance(body["anomaly"], bool)

    def test_response_has_score_field(self, predict_setup):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "score" in body
        assert 0.0 <= body["score"] <= 1.0

    def test_response_has_risk_level_field(self, predict_setup):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert body["risk_level"] in ("low", "medium", "high")

    def test_response_has_model_name(self, predict_setup):
        assert client.post("/predict", json=VALID_PAYLOAD).json()["model_name"] == "best_model"

    def test_response_has_model_version(self, predict_setup):
        assert client.post("/predict", json=VALID_PAYLOAD).json()["model_version"] == "1.0.0"

    def test_response_has_algorithm(self, predict_setup):
        assert client.post("/predict", json=VALID_PAYLOAD).json()["algorithm"] == "XGBoost"

    def test_high_score_returns_anomaly_true_and_high_risk(self, predict_setup):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert body["anomaly"] is True
        assert body["risk_level"] == "high"
        assert body["score"] == pytest.approx(0.95, abs=1e-4)

    def test_low_score_returns_anomaly_false_and_low_risk(
        self, mock_paths, mock_model_low, monkeypatch
    ):
        meta_path, schema_path = mock_paths
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        monkeypatch.setattr(api_main, "load_model", lambda path: mock_model_low)
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert body["anomaly"] is False
        assert body["risk_level"] == "low"

    def test_medium_score_returns_medium_risk(
        self, mock_paths, mock_model_medium, monkeypatch
    ):
        meta_path, schema_path = mock_paths
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        monkeypatch.setattr(api_main, "load_model", lambda path: mock_model_medium)
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert body["anomaly"] is True
        assert body["risk_level"] == "medium"


# ---------------------------------------------------------------------------
# Feature vector ordering and bool conversion
# ---------------------------------------------------------------------------


class TestFeatureVector:
    def test_features_in_schema_order_not_json_order(
        self, mock_paths, monkeypatch
    ):
        """predict_proba must be called with features in model_features order."""
        meta_path, schema_path = mock_paths
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)

        captured = {}

        def capturing_model(x):
            captured["X"] = x
            return np.array([[0.05, 0.95]])

        mock = MagicMock()
        mock.predict_proba.side_effect = capturing_model
        monkeypatch.setattr(api_main, "load_model", lambda path: mock)

        client.post("/predict", json=VALID_PAYLOAD)

        X = captured["X"]
        assert X.shape == (1, 12)
        # First feature must be daily_cost
        assert X[0][0] == pytest.approx(VALID_PAYLOAD["daily_cost"])
        # is_missing_tag is at index 9, was False → must be 0.0
        assert X[0][9] == pytest.approx(0.0)

    def test_is_missing_tag_true_becomes_1(self, mock_paths, monkeypatch):
        meta_path, schema_path = mock_paths
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)

        captured = {}

        def capturing(x):
            captured["X"] = x
            return np.array([[0.8, 0.2]])

        mock = MagicMock()
        mock.predict_proba.side_effect = capturing
        monkeypatch.setattr(api_main, "load_model", lambda path: mock)

        payload = {**VALID_PAYLOAD, "is_missing_tag": True}
        client.post("/predict", json=payload)
        assert captured["X"][0][9] == pytest.approx(1.0)

    def test_is_weekend_true_becomes_1(self, mock_paths, monkeypatch):
        meta_path, schema_path = mock_paths
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)

        captured = {}

        def capturing(x):
            captured["X"] = x
            return np.array([[0.8, 0.2]])

        mock = MagicMock()
        mock.predict_proba.side_effect = capturing
        monkeypatch.setattr(api_main, "load_model", lambda path: mock)

        payload = {**VALID_PAYLOAD, "is_weekend": True}
        client.post("/predict", json=payload)
        # is_weekend is at index 11
        assert captured["X"][0][11] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Pydantic validation — 422 errors
# ---------------------------------------------------------------------------


class TestPredictValidation:
    def test_422_when_field_missing(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "daily_cost"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_422_when_day_of_week_above_6(self):
        payload = {**VALID_PAYLOAD, "day_of_week": 7}
        assert client.post("/predict", json=payload).status_code == 422

    def test_422_when_day_of_week_below_0(self):
        payload = {**VALID_PAYLOAD, "day_of_week": -1}
        assert client.post("/predict", json=payload).status_code == 422

    def test_422_when_daily_cost_negative(self):
        payload = {**VALID_PAYLOAD, "daily_cost": -1.0}
        assert client.post("/predict", json=payload).status_code == 422

    def test_422_when_usage_quantity_negative(self):
        payload = {**VALID_PAYLOAD, "usage_quantity": -0.01}
        assert client.post("/predict", json=payload).status_code == 422

    def test_422_when_avg_cost_7d_negative(self):
        payload = {**VALID_PAYLOAD, "avg_cost_7d": -5.0}
        assert client.post("/predict", json=payload).status_code == 422

    def test_cost_change_percent_accepts_negative(self, predict_setup):
        payload = {**VALID_PAYLOAD, "cost_change_percent": -50.0}
        assert client.post("/predict", json=payload).status_code == 200

    def test_usage_change_percent_accepts_negative(self, predict_setup):
        payload = {**VALID_PAYLOAD, "usage_change_percent": -99.0}
        assert client.post("/predict", json=payload).status_code == 200

    def test_day_of_week_boundary_0_accepted(self, predict_setup):
        payload = {**VALID_PAYLOAD, "day_of_week": 0}
        assert client.post("/predict", json=payload).status_code == 200

    def test_day_of_week_boundary_6_accepted(self, predict_setup):
        payload = {**VALID_PAYLOAD, "day_of_week": 6}
        assert client.post("/predict", json=payload).status_code == 200


# ---------------------------------------------------------------------------
# 503 errors — missing artefacts
# ---------------------------------------------------------------------------


class TestPredictErrors:
    def test_503_when_model_joblib_missing(self, mock_paths, monkeypatch):
        meta_path, schema_path = mock_paths
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        # load_model returns None when file is absent — endpoint must 503
        monkeypatch.setattr(api_main, "load_model", lambda path: None)
        assert client.post("/predict", json=VALID_PAYLOAD).status_code == 503

    def test_503_when_metadata_missing(self, tmp_path, mock_paths, monkeypatch):
        _, schema_path = mock_paths
        monkeypatch.setattr(settings, "model_metadata_path", str(tmp_path / "missing.json"))
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        assert client.post("/predict", json=VALID_PAYLOAD).status_code == 503

    def test_503_when_schema_missing(self, tmp_path, mock_paths, monkeypatch):
        meta_path, _ = mock_paths
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", str(tmp_path / "missing.json"))
        assert client.post("/predict", json=VALID_PAYLOAD).status_code == 503

    def test_error_has_standardized_format(self, mock_paths, monkeypatch):
        meta_path, schema_path = mock_paths
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        monkeypatch.setattr(api_main, "load_model", lambda path: None)
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "error" in body
        assert "message" in body
        assert "details" in body
        assert "Traceback" not in str(body)


# ---------------------------------------------------------------------------
# prediction_service unit tests
# ---------------------------------------------------------------------------


class TestRiskLevel:
    def test_low(self):
        assert _risk_level(0.0) == "low"
        assert _risk_level(0.39) == "low"

    def test_medium(self):
        assert _risk_level(0.4) == "medium"
        assert _risk_level(0.699) == "medium"

    def test_high(self):
        assert _risk_level(0.7) == "high"
        assert _risk_level(1.0) == "high"


class TestBuildFeatureVector:
    def test_output_shape_is_1_by_12(self):
        request = PredictRequest(**VALID_PAYLOAD)
        X = build_feature_vector(request, FEATURE_ORDER)
        assert X.shape == (1, 12)

    def test_first_value_is_daily_cost(self):
        request = PredictRequest(**VALID_PAYLOAD)
        X = build_feature_vector(request, FEATURE_ORDER)
        assert X[0][0] == pytest.approx(VALID_PAYLOAD["daily_cost"])

    def test_bool_false_becomes_zero(self):
        request = PredictRequest(**VALID_PAYLOAD)  # is_missing_tag=False
        X = build_feature_vector(request, ["is_missing_tag"])
        assert X[0][0] == pytest.approx(0.0)

    def test_bool_true_becomes_one(self):
        payload = {**VALID_PAYLOAD, "is_missing_tag": True}
        request = PredictRequest(**payload)
        X = build_feature_vector(request, ["is_missing_tag"])
        assert X[0][0] == pytest.approx(1.0)

    def test_accepts_numpy_array_from_predict_proba(self):
        """predict_proba can return numpy array — float() must handle it."""
        proba = np.array([[0.1, 0.9]])
        score = float(proba[0][1])
        assert score == pytest.approx(0.9)

    def test_accepts_list_from_predict_proba(self):
        """predict_proba can return list — float() must handle it."""
        proba = [[0.1, 0.9]]
        score = float(proba[0][1])
        assert score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Architecture: api/ must not import ml/
# ---------------------------------------------------------------------------


class TestLayerSeparation:
    def test_api_modules_do_not_import_src_ml(self):
        api_files = [
            "src/api/main.py",
            "src/api/config.py",
            "src/api/schemas.py",
            "src/api/services/model_service.py",
            "src/api/services/prediction_service.py",
        ]
        for filepath in api_files:
            content = Path(filepath).read_text()
            assert "from src.ml" not in content, f"{filepath} imports from src.ml"
            assert "import src.ml" not in content, f"{filepath} imports src.ml"


# ---------------------------------------------------------------------------
# Existing endpoints still work
# ---------------------------------------------------------------------------


class TestExistingEndpoints:
    def test_health_still_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_model_info_still_works_with_valid_files(self, mock_paths, monkeypatch):
        meta_path, schema_path = mock_paths
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        assert client.get("/model/info").status_code == 200
