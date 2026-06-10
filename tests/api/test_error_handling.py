"""Tests for standardised error responses and OpenAPI documentation."""

import json

import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main
from src.api.config import settings
from src.api.main import app

client = TestClient(app)

MOCK_METADATA = {
    "model_name": "best_model",
    "model_version": "1.0.0",
    "algorithm": "XGBoost",
    "model_type": "supervised_ml",
    "features": ["daily_cost", "usage_quantity"],
    "target": "is_anomaly",
    "evaluation_scope": "final_model_trained_on_full_dataset_for_serving",
}

MOCK_SCHEMA = {"feature_count": 12, "model_features": ["daily_cost", "usage_quantity"]}

VALID_PAYLOAD = {
    "daily_cost": 100.0, "usage_quantity": 500.0,
    "previous_day_cost": 90.0, "previous_day_usage": 480.0,
    "avg_cost_7d": 95.0, "avg_cost_30d": 92.0,
    "cost_change_percent": 11.0, "usage_change_percent": 4.0,
    "cost_to_usage_ratio": 0.2, "is_missing_tag": False,
    "day_of_week": 1, "is_weekend": False,
}


@pytest.fixture
def mock_paths(tmp_path):
    meta = tmp_path / "metadata.json"
    schema = tmp_path / "schema.json"
    meta.write_text(json.dumps(MOCK_METADATA))
    schema.write_text(json.dumps(MOCK_SCHEMA))
    return str(meta), str(schema)


# ---------------------------------------------------------------------------
# Standardised error format
# ---------------------------------------------------------------------------


class TestErrorFormat:
    def test_model_info_503_has_error_field(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "model_metadata_path", str(tmp_path / "x.json"))
        monkeypatch.setattr(settings, "feature_schema_path", str(tmp_path / "x.json"))
        body = client.get("/model/info").json()
        assert "error" in body

    def test_model_info_503_has_message_field(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "model_metadata_path", str(tmp_path / "x.json"))
        monkeypatch.setattr(settings, "feature_schema_path", str(tmp_path / "x.json"))
        body = client.get("/model/info").json()
        assert "message" in body

    def test_model_info_503_has_details_field(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "model_metadata_path", str(tmp_path / "x.json"))
        monkeypatch.setattr(settings, "feature_schema_path", str(tmp_path / "x.json"))
        body = client.get("/model/info").json()
        assert "details" in body

    def test_model_info_503_error_code_is_artifact_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "model_metadata_path", str(tmp_path / "x.json"))
        monkeypatch.setattr(settings, "feature_schema_path", str(tmp_path / "x.json"))
        body = client.get("/model/info").json()
        assert body["error"] == "model_artifact_unavailable"

    def test_model_info_503_no_traceback_in_body(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "model_metadata_path", str(tmp_path / "x.json"))
        monkeypatch.setattr(settings, "feature_schema_path", str(tmp_path / "x.json"))
        body = client.get("/model/info").json()
        assert "Traceback" not in str(body)
        assert "Exception" not in str(body)

    def test_predict_503_has_error_field(self, mock_paths, monkeypatch):
        meta_path, schema_path = mock_paths
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        monkeypatch.setattr(api_main, "load_model", lambda path: None)
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "error" in body

    def test_predict_503_has_message_field(self, mock_paths, monkeypatch):
        meta_path, schema_path = mock_paths
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        monkeypatch.setattr(api_main, "load_model", lambda path: None)
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "message" in body

    def test_predict_503_has_details_field(self, mock_paths, monkeypatch):
        meta_path, schema_path = mock_paths
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        monkeypatch.setattr(api_main, "load_model", lambda path: None)
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "details" in body

    def test_predict_503_error_code_is_artifact_unavailable(self, mock_paths, monkeypatch):
        meta_path, schema_path = mock_paths
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        monkeypatch.setattr(api_main, "load_model", lambda path: None)
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert body["error"] == "model_artifact_unavailable"

    def test_error_format_consistent_across_endpoints(self, tmp_path, mock_paths, monkeypatch):
        """Both /model/info and /predict 503 errors share the same schema."""
        monkeypatch.setattr(settings, "model_metadata_path", str(tmp_path / "x.json"))
        monkeypatch.setattr(settings, "feature_schema_path", str(tmp_path / "x.json"))
        monkeypatch.setattr(api_main, "load_model", lambda path: None)

        body_info = client.get("/model/info").json()
        body_pred = client.post("/predict", json=VALID_PAYLOAD).json()

        for body in (body_info, body_pred):
            assert set(body.keys()) >= {"error", "message", "details"}
            assert body["error"] == "model_artifact_unavailable"


# ---------------------------------------------------------------------------
# OpenAPI specification
# ---------------------------------------------------------------------------


class TestOpenAPI:
    @pytest.fixture(autouse=True)
    def openapi(self):
        return client.get("/openapi.json").json()

    def test_openapi_endpoint_returns_200(self):
        assert client.get("/openapi.json").status_code == 200

    def test_title_correct(self, openapi):
        assert openapi["info"]["title"] == "Cloud Cost Anomaly Detection API"

    def test_description_present(self, openapi):
        assert len(openapi["info"]["description"]) > 10

    def test_version_present(self, openapi):
        assert "version" in openapi["info"]

    def test_health_path_exists(self, openapi):
        assert "/health" in openapi["paths"]

    def test_model_info_path_exists(self, openapi):
        assert "/model/info" in openapi["paths"]

    def test_predict_path_exists(self, openapi):
        assert "/predict" in openapi["paths"]

    def test_health_has_tag(self, openapi):
        tags = openapi["paths"]["/health"]["get"].get("tags", [])
        assert "health" in tags

    def test_model_info_has_tag(self, openapi):
        tags = openapi["paths"]["/model/info"]["get"].get("tags", [])
        assert "model" in tags

    def test_predict_has_tag(self, openapi):
        tags = openapi["paths"]["/predict"]["post"].get("tags", [])
        assert "prediction" in tags

    def test_predict_has_summary(self, openapi):
        summary = openapi["paths"]["/predict"]["post"].get("summary", "")
        assert len(summary) > 5

    def test_503_documented_for_model_info(self, openapi):
        responses = openapi["paths"]["/model/info"]["get"]["responses"]
        assert "503" in responses

    def test_503_documented_for_predict(self, openapi):
        responses = openapi["paths"]["/predict"]["post"]["responses"]
        assert "503" in responses

    def test_422_documented_for_predict(self, openapi):
        responses = openapi["paths"]["/predict"]["post"]["responses"]
        assert "422" in responses


# ---------------------------------------------------------------------------
# Layer separation (no src/ml import in src/api/)
# ---------------------------------------------------------------------------


class TestLayerSeparation:
    def test_api_files_do_not_import_src_ml(self):
        from pathlib import Path
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
# Existing endpoints still work after error handling changes
# ---------------------------------------------------------------------------


class TestRegressionAfterErrorHandling:
    def test_health_still_200(self):
        assert client.get("/health").status_code == 200

    def test_health_still_returns_ok(self):
        assert client.get("/health").json()["status"] == "ok"

    def test_model_info_still_200_when_files_present(self, mock_paths, monkeypatch):
        meta_path, schema_path = mock_paths
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        assert client.get("/model/info").status_code == 200
