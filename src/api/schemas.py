"""Pydantic v2 request/response schemas for the Cloud Cost Anomaly Detection API."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ModelInfoResponse(BaseModel):
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
    before calling this endpoint. See feature_schema.json for the full contract.
    """

    daily_cost: float = Field(ge=0.0, description="Daily cost in USD")
    usage_quantity: float = Field(ge=0.0, description="Daily usage quantity")
    previous_day_cost: float = Field(ge=0.0, description="Cost the previous day")
    previous_day_usage: float = Field(ge=0.0, description="Usage the previous day")
    avg_cost_7d: float = Field(ge=0.0, description="7-day rolling average cost")
    avg_cost_30d: float = Field(ge=0.0, description="30-day rolling average cost")
    cost_change_percent: float = Field(description="Day-over-day cost change %")
    usage_change_percent: float = Field(description="Day-over-day usage change %")
    cost_to_usage_ratio: float = Field(ge=0.0, description="Cost divided by usage")
    is_missing_tag: bool = Field(description="True when tag_project or tag_owner is empty")
    day_of_week: int = Field(ge=0, le=6, description="0=Monday … 6=Sunday")
    is_weekend: bool = Field(description="True when day_of_week >= 5")


class PredictResponse(BaseModel):
    anomaly: bool
    score: float = Field(ge=0.0, le=1.0, description="Anomaly probability [0, 1]")
    risk_level: Literal["low", "medium", "high"]
    model_name: str
    model_version: str
    algorithm: str
