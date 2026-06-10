"""Cloud Cost Anomaly Detection API — Phase 2.1 skeleton.

Endpoints:
  GET /health      — liveness check, always 200 regardless of model files
  GET /model/info  — model metadata and feature schema (503 if files missing)

Run locally:
  uvicorn src.api.main:app --reload
"""

from fastapi import FastAPI, HTTPException

from src.api.config import settings
from src.api.schemas import HealthResponse, ModelInfoResponse
from src.api.services.model_service import load_feature_schema, load_metadata

app = FastAPI(
    title="Cloud Cost Anomaly Detection API",
    description="FinOps + MLOps anomaly detection platform — serving layer.",
    version=settings.api_version,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check.

    Returns 200 regardless of whether model artefacts are present on disk.
    """
    return HealthResponse(
        status="ok",
        service="cloud-cost-anomaly-api",
        version=settings.api_version,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return model metadata and feature contract.

    Reads model_metadata.json and feature_schema.json from the paths
    configured via MODEL_METADATA_PATH and FEATURE_SCHEMA_PATH.

    Returns HTTP 503 with a descriptive message if either file is missing.
    Does NOT require best_model.joblib to be present.
    """
    try:
        metadata = load_metadata(settings.model_metadata_path)
        schema = load_feature_schema(settings.feature_schema_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return ModelInfoResponse(
        model_name=metadata["model_name"],
        model_version=metadata["model_version"],
        algorithm=metadata["algorithm"],
        model_type=metadata["model_type"],
        feature_count=schema["feature_count"],
        features=metadata["features"],
        target=metadata["target"],
        evaluation_scope=metadata["evaluation_scope"],
    )
