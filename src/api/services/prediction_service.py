"""Prediction logic for the anomaly detection API.

Pure functions — no I/O, no model loading, no file access.
Receives a fitted model and pre-loaded artefact dicts from the endpoint layer.

This module does NOT import from src/ml/. It operates on pre-generated
artefacts (model, metadata, schema) loaded by model_service.py.
"""

from typing import Any, Literal

import numpy as np

from src.api.schemas import PredictRequest, PredictResponse


def _risk_level(score: float) -> Literal["low", "medium", "high"]:
    """Map a probability score to a human-readable risk level."""
    if score < 0.4:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def build_feature_vector(
    request: PredictRequest,
    feature_order: list[str],
) -> np.ndarray:
    """Build a 2D feature matrix (shape 1 × len(feature_order)) for inference.

    feature_order comes from feature_schema.json["model_features"], guaranteeing
    the correct column order regardless of JSON key order in the request.

    Boolean fields (is_missing_tag, is_weekend) are converted to int (0 or 1).

    Raises:
        ValueError: if feature_order references a feature not present in the request.
    """
    raw = request.model_dump()
    try:
        values = [
            float(int(raw[feat])) if isinstance(raw[feat], bool) else float(raw[feat])
            for feat in feature_order
        ]
    except KeyError as exc:
        raise ValueError(f"Feature {exc} from schema not found in request") from exc
    return np.array([values], dtype=float)


def predict(
    request: PredictRequest,
    model: Any,
    metadata: dict,
    schema: dict,
) -> PredictResponse:
    """Run inference on the fitted model and return a PredictResponse.

    Args:
        request:  Validated PredictRequest with the 12 model features.
        model:    Fitted XGBClassifier (or any object with predict_proba).
        metadata: Contents of model_metadata.json.
        schema:   Contents of feature_schema.json.

    Returns:
        PredictResponse with anomaly flag, score, risk_level, and model info.
    """
    feature_order: list[str] = schema.get("model_features") or schema.get(
        "required_model_features", []
    )
    X = build_feature_vector(request, feature_order)

    proba = model.predict_proba(X)
    # Works for both numpy ndarray and list return types.
    score = float(proba[0][1])

    return PredictResponse(
        anomaly=score >= 0.5,
        score=round(score, 4),
        risk_level=_risk_level(score),
        model_name=metadata["model_name"],
        model_version=metadata["model_version"],
        algorithm=metadata["algorithm"],
    )
