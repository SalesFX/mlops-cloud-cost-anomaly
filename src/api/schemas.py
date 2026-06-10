"""Pydantic v2 response schemas for the Cloud Cost Anomaly Detection API."""

from pydantic import BaseModel


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
