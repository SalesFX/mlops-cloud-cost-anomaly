"""Model artefact loading service.

Handles reading model_metadata.json, feature_schema.json, and
best_model.joblib from the local models/ directory.

This module does NOT import from src/ml/ — it loads pre-generated artefacts
from disk, keeping the API layer independent of the ML training layer (ADR-004).
"""

import json
from pathlib import Path
from typing import Any

import joblib


def load_json(path: str) -> dict:
    """Load a JSON file from disk.

    Raises FileNotFoundError with a clear message if the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(p) as f:
        return json.load(f)


def load_metadata(path: str) -> dict:
    """Load model_metadata.json."""
    return load_json(path)


def load_feature_schema(path: str) -> dict:
    """Load feature_schema.json."""
    return load_json(path)


def load_model(path: str) -> Any:
    """Load a joblib-serialised model from disk.

    Returns None if the file does not exist — the API remains operational
    for /health and /model/info even when the model artefact is absent.
    """
    p = Path(path)
    if not p.exists():
        return None
    return joblib.load(p)
