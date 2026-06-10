"""Tests for GET /health and GET /model/info endpoints.

Uses FastAPI TestClient + monkeypatch to override settings paths so tests
never depend on artefacts in models/ being present on disk.
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.api.config import settings
from src.api.main import app
from src.api.services.model_service import load_json, load_model

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_metadata(tmp_path) -> tuple[str, dict]:
    data = {
        "model_name": "best_model",
        "model_version": "1.0.0",
        "algorithm": "XGBoost",
        "model_type": "supervised_ml",
        "features": ["daily_cost", "usage_quantity", "avg_cost_7d"],
        "target": "is_anomaly",
        "evaluation_scope": "final_model_trained_on_full_dataset_for_serving",
    }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(data))
    return str(path), data


@pytest.fixture
def mock_schema(tmp_path) -> tuple[str, dict]:
    data = {"feature_count": 12, "model_features": ["daily_cost", "usage_quantity"]}
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(data))
    return str(path), data


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_status_is_ok(self):
        assert client.get("/health").json()["status"] == "ok"

    def test_service_name_correct(self):
        assert client.get("/health").json()["service"] == "cloud-cost-anomaly-api"

    def test_version_field_present(self):
        assert "version" in client.get("/health").json()

    def test_health_independent_of_model_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "model_path", str(tmp_path / "no.joblib"))
        monkeypatch.setattr(settings, "model_metadata_path", str(tmp_path / "no.json"))
        monkeypatch.setattr(settings, "feature_schema_path", str(tmp_path / "no.json"))
        assert client.get("/health").status_code == 200

    def test_response_schema_has_all_fields(self):
        body = client.get("/health").json()
        for key in ("status", "service", "version"):
            assert key in body


# ---------------------------------------------------------------------------
# GET /model/info
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_returns_200_when_files_exist(self, mock_metadata, mock_schema, monkeypatch):
        meta_path, _ = mock_metadata
        schema_path, _ = mock_schema
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        assert client.get("/model/info").status_code == 200

    def test_returns_model_name(self, mock_metadata, mock_schema, monkeypatch):
        meta_path, _ = mock_metadata
        schema_path, _ = mock_schema
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        assert client.get("/model/info").json()["model_name"] == "best_model"

    def test_returns_model_version(self, mock_metadata, mock_schema, monkeypatch):
        meta_path, _ = mock_metadata
        schema_path, _ = mock_schema
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        assert client.get("/model/info").json()["model_version"] == "1.0.0"

    def test_returns_algorithm(self, mock_metadata, mock_schema, monkeypatch):
        meta_path, _ = mock_metadata
        schema_path, _ = mock_schema
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        assert client.get("/model/info").json()["algorithm"] == "XGBoost"

    def test_returns_feature_count(self, mock_metadata, mock_schema, monkeypatch):
        meta_path, _ = mock_metadata
        schema_path, _ = mock_schema
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        assert client.get("/model/info").json()["feature_count"] == 12

    def test_returns_features_list(self, mock_metadata, mock_schema, monkeypatch):
        meta_path, _ = mock_metadata
        schema_path, _ = mock_schema
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        body = client.get("/model/info").json()
        assert isinstance(body["features"], list)
        assert len(body["features"]) > 0

    def test_returns_evaluation_scope(self, mock_metadata, mock_schema, monkeypatch):
        meta_path, _ = mock_metadata
        schema_path, _ = mock_schema
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        scope = client.get("/model/info").json()["evaluation_scope"]
        assert scope == "final_model_trained_on_full_dataset_for_serving"

    def test_503_when_metadata_missing(self, tmp_path, mock_schema, monkeypatch):
        schema_path, _ = mock_schema
        monkeypatch.setattr(settings, "model_metadata_path", str(tmp_path / "missing.json"))
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        assert client.get("/model/info").status_code == 503

    def test_503_when_schema_missing(self, mock_metadata, tmp_path, monkeypatch):
        meta_path, _ = mock_metadata
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", str(tmp_path / "missing.json"))
        assert client.get("/model/info").status_code == 503

    def test_error_has_detail_not_traceback(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "model_metadata_path", str(tmp_path / "missing.json"))
        monkeypatch.setattr(settings, "feature_schema_path", str(tmp_path / "missing.json"))
        body = client.get("/model/info").json()
        assert "detail" in body
        assert "Traceback" not in str(body.get("detail", ""))

    def test_model_info_independent_of_joblib_file(self, mock_metadata, mock_schema,
                                                    tmp_path, monkeypatch):
        meta_path, _ = mock_metadata
        schema_path, _ = mock_schema
        monkeypatch.setattr(settings, "model_metadata_path", meta_path)
        monkeypatch.setattr(settings, "feature_schema_path", schema_path)
        monkeypatch.setattr(settings, "model_path", str(tmp_path / "no_model.joblib"))
        assert client.get("/model/info").status_code == 200


# ---------------------------------------------------------------------------
# model_service unit tests
# ---------------------------------------------------------------------------


class TestModelService:
    def test_load_json_returns_dict(self, tmp_path):
        data = {"key": "value", "count": 42}
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))
        assert load_json(str(path)) == data

    def test_load_json_preserves_nested_structures(self, tmp_path):
        data = {"features": ["a", "b"], "nested": {"x": 1}}
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))
        assert load_json(str(path)) == data

    def test_load_json_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_json(str(tmp_path / "nonexistent.json"))

    def test_load_model_returns_none_when_missing(self, tmp_path):
        assert load_model(str(tmp_path / "nonexistent.joblib")) is None

    def test_load_model_returns_object_when_file_exists(self, tmp_path):
        import joblib
        data = {"model": "mock", "version": 1}
        path = tmp_path / "mock.joblib"
        joblib.dump(data, str(path))
        loaded = load_model(str(path))
        assert loaded == data
