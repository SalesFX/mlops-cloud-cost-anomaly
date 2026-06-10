"""Cloud Cost Anomaly Detection API.

Endpoints:
  GET  /health      — liveness check, always 200 regardless of model files
  GET  /model/info  — model metadata and feature schema (503 if files missing)
  POST /predict     — anomaly detection on pre-engineered billing features

Run locally:
  uvicorn src.api.main:app --reload
"""

from fastapi import FastAPI, HTTPException

from src.api.config import settings
from src.api.schemas import HealthResponse, ModelInfoResponse, PredictRequest, PredictResponse
from src.api.services.model_service import load_feature_schema, load_metadata, load_model
from src.api.services.prediction_service import predict as run_prediction

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


@app.post("/predict", response_model=PredictResponse)
def predict_anomaly(request: PredictRequest) -> PredictResponse:
    """Detect anomalies in pre-engineered billing features.

    Receives the 12 model features from feature_schema.json.
    Features must be pre-computed (run feature_engineering.py first).
    Returns anomaly flag, probability score, risk level, and model provenance.

    Returns HTTP 503 if any model artefact is missing from disk.
    """
    try:
        metadata = load_metadata(settings.model_metadata_path)
        schema = load_feature_schema(settings.feature_schema_path)
        model = load_model(settings.model_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model artefact not found: {settings.model_path}",
        )

    return run_prediction(request, model, metadata, schema)
