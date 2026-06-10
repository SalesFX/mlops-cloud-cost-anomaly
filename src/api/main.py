"""Cloud Cost Anomaly Detection API.

Endpoints:
  GET  /health      — liveness check, always 200 regardless of model files
  GET  /model/info  — model metadata and feature schema (503 if files missing)
  POST /predict     — anomaly detection on pre-engineered billing features

Run locally:
  uvicorn src.api.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from starlette.requests import Request

from src.api.config import settings
from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from src.api.services.model_service import load_feature_schema, load_metadata, load_model
from src.api.services.prediction_service import predict as run_prediction

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cloud Cost Anomaly Detection API",
    description=(
        "API for serving an XGBoost model that detects anomalies in "
        "engineered cloud cost features. Part of the FinOps + MLOps platform."
    ),
    version=settings.api_version,
)

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

_ERROR_503 = {503: {"model": ErrorResponse, "description": "Model artefact unavailable"}}


@app.exception_handler(HTTPException)
async def standardized_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return a flat ErrorResponse body when the detail is a structured error dict.

    All other HTTPException responses fall through to FastAPI's default handler,
    which wraps them as {"detail": "..."}.
    """
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return await http_exception_handler(request, exc)


def _artifact_error(path: str) -> dict:
    """Build the standard 503 error body for a missing model artefact."""
    return {
        "error": "model_artifact_unavailable",
        "message": "Required model artefact file not found.",
        "details": path,
    }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Liveness check",
)
def health() -> HealthResponse:
    """Return 200 regardless of whether model artefacts are present on disk.

    Use this endpoint to verify the API process is running before calling
    `/model/info` or `/predict`.
    """
    return HealthResponse(
        status="ok",
        service="cloud-cost-anomaly-api",
        version=settings.api_version,
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    responses=_ERROR_503,
    tags=["model"],
    summary="Model metadata and feature contract",
)
def model_info() -> ModelInfoResponse:
    """Return provenance and feature schema for the loaded model.

    Reads `model_metadata.json` and `feature_schema.json` from the paths
    configured via `MODEL_METADATA_PATH` and `FEATURE_SCHEMA_PATH`.

    Does **not** require `best_model.joblib` to be present.
    Returns **HTTP 503** with a structured error if either JSON file is missing.
    """
    try:
        metadata = load_metadata(settings.model_metadata_path)
        schema = load_feature_schema(settings.feature_schema_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=_artifact_error(str(exc)))

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


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={**_ERROR_503, 422: {"description": "Validation error — invalid or missing field"}},
    tags=["prediction"],
    summary="Anomaly detection on engineered billing features",
)
def predict_anomaly(request: PredictRequest) -> PredictResponse:
    """Detect anomalies in a set of pre-engineered cloud cost features.

    **Input:** the 12 model features defined in `feature_schema.json`.
    Features must be pre-computed (rolling averages, change percents, ratio, flags).
    This endpoint does **not** accept raw billing records and does **not** run
    feature engineering internally.

    **Output:** anomaly flag, probability score [0–1], risk level, and model provenance.

    Returns **HTTP 422** for invalid or missing fields.
    Returns **HTTP 503** if any model artefact is missing from disk.
    """
    try:
        metadata = load_metadata(settings.model_metadata_path)
        schema = load_feature_schema(settings.feature_schema_path)
        model = load_model(settings.model_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=_artifact_error(str(exc)))

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=_artifact_error(settings.model_path),
        )

    return run_prediction(request, model, metadata, schema)
