# src/api — API Layer

FastAPI application for the Cloud Cost Anomaly Detection Platform.  
Phase 2.1: Skeleton + model loading. Prediction endpoint added in Phase 2.2.

---

## Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate the model artefacts from Phase 1.8 if not already present:

```bash
python src/ml/model_registry.py \
    --input data/cloud_cost_features.csv \
    --model-output models/best_model.joblib \
    --metadata-output models/model_metadata.json \
    --schema-output models/feature_schema.json \
    --model-version 1.0.0 \
    --seed 42
```

---

## Running locally

```bash
uvicorn src.api.main:app --reload
```

| URL | Description |
|-----|-------------|
| `http://localhost:8000/health` | Liveness check |
| `http://localhost:8000/model/info` | Model metadata and feature schema |
| `http://localhost:8000/docs` | Swagger UI |
| `http://localhost:8000/redoc` | ReDoc |

---

## Endpoints

### GET /health

Always returns 200 regardless of model artefact availability.

```json
{
  "status": "ok",
  "service": "cloud-cost-anomaly-api",
  "version": "0.1.0"
}
```

### GET /model/info

Returns model metadata from `models/model_metadata.json` and feature count
from `models/feature_schema.json`. Returns **HTTP 503** if either file is missing.
Does **not** require `models/best_model.joblib` to be present.

```json
{
  "model_name": "best_model",
  "model_version": "1.0.0",
  "algorithm": "XGBoost",
  "model_type": "supervised_ml",
  "feature_count": 12,
  "features": ["daily_cost", "usage_quantity", ...],
  "target": "is_anomaly",
  "evaluation_scope": "final_model_trained_on_full_dataset_for_serving"
}
```

---

## Configuration

Set via environment variables (defaults shown):

```bash
export MODEL_PATH=models/best_model.joblib
export MODEL_METADATA_PATH=models/model_metadata.json
export FEATURE_SCHEMA_PATH=models/feature_schema.json
export API_VERSION=0.1.0
```

---

## Module structure

```
src/api/
├── __init__.py
├── config.py             # Settings from env vars, singleton `settings`
├── schemas.py            # Pydantic v2 response models
├── services/
│   ├── __init__.py
│   └── model_service.py  # load_json, load_metadata, load_feature_schema, load_model
└── main.py               # FastAPI app, GET /health, GET /model/info
```

**Architecture note (ADR-004):** `src/api/` does not import from `src/ml/`.
It loads pre-generated artefacts from `models/` via `model_service.py`.

---

## Running tests

```bash
pytest tests/api/ -v
```

---

## Upcoming (Phase 2.2+)

- `POST /predict` — receive a billing record, return anomaly prediction
- Pydantic v2 input validation for prediction request
- `GET /model/info` extended with serving contract details
