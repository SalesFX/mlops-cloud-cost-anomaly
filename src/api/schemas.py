"""Pydantic v2 request/response schemas for the Cloud Cost Anomaly Detection API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "service": "cloud-cost-anomaly-api",
                "version": "0.1.0",
            }
        }
    )

    status: str
    service: str
    version: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "best_model",
                "model_version": "1.0.0",
                "algorithm": "XGBoost",
                "model_type": "supervised_ml",
                "feature_count": 12,
                "features": [
                    "daily_cost", "usage_quantity", "previous_day_cost",
                    "previous_day_usage", "avg_cost_7d", "avg_cost_30d",
                    "cost_change_percent", "usage_change_percent",
                    "cost_to_usage_ratio", "is_missing_tag", "day_of_week", "is_weekend",
                ],
                "target": "is_anomaly",
                "evaluation_scope": "final_model_trained_on_full_dataset_for_serving",
            }
        }
    )

    model_name: str
    model_version: str
    algorithm: str
    model_type: str
    feature_count: int
    features: list[str]
    target: str
    evaluation_scope: str


class PredictRequest(BaseModel):
    """The 12 engineered model features expected by best_model.joblib.

    Features must be pre-computed (rolling averages, change percents, etc.)
    before calling this endpoint. The API does **not** accept raw billing records
    and does **not** perform feature engineering internally.
    See feature_schema.json for the full feature contract.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    daily_cost: float = Field(ge=0.0, description="Daily cost in USD (≥ 0)")
    usage_quantity: float = Field(ge=0.0, description="Daily usage quantity (≥ 0)")
    previous_day_cost: float = Field(ge=0.0, description="Cost the previous day (≥ 0)")
    previous_day_usage: float = Field(ge=0.0, description="Usage the previous day (≥ 0)")
    avg_cost_7d: float = Field(ge=0.0, description="7-day rolling average cost (≥ 0)")
    avg_cost_30d: float = Field(ge=0.0, description="30-day rolling average cost (≥ 0)")
    cost_change_percent: float = Field(description="Day-over-day cost change % (can be negative)")
    usage_change_percent: float = Field(description="Day-over-day usage change % (can be negative)")
    cost_to_usage_ratio: float = Field(ge=0.0, description="daily_cost / usage_quantity (≥ 0)")
    is_missing_tag: bool = Field(description="True when tag_project or tag_owner is empty")
    day_of_week: int = Field(ge=0, le=6, description="0=Monday … 6=Sunday")
    is_weekend: bool = Field(description="True when day_of_week >= 5")


class PredictResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "anomaly": True,
                "score": 0.9995,
                "risk_level": "high",
                "model_name": "best_model",
                "model_version": "1.0.0",
                "algorithm": "XGBoost",
            }
        }
    )

    anomaly: bool = Field(description="True when score ≥ 0.5")
    score: float = Field(ge=0.0, le=1.0, description="Anomaly probability [0, 1]")
    risk_level: Literal["low", "medium", "high"] = Field(
        description="low (<0.4) / medium (0.4–0.7) / high (≥0.7)"
    )
    model_name: str
    model_version: str
    algorithm: str


class ErrorResponse(BaseModel):
    """Standardised error body returned for HTTP 503 artefact-unavailable errors."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "model_artifact_unavailable",
                "message": "Required model artefact file not found.",
                "details": "models/model_metadata.json",
            }
        }
    )

    error: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable description")
    details: str = Field(description="Specific path or context that triggered the error")
