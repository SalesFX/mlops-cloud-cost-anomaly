# Cloud Cost Anomaly Detection API
# Multi-stage build: builder installs deps, runner is the minimal production image.
#
# Model artefacts (models/) are NOT baked into this image.
# They must be provided at runtime via volume mount:
#
#   docker run --rm -p 8000:8000 \
#     -v "$(pwd)/models:/app/models:ro" \
#     cloud-cost-anomaly-api:local
#
# Without a volume mount, GET /health returns 200 and
# GET /model/info + POST /predict return HTTP 503 (controlled).

# ---------------------------------------------------------------------------
# Stage 1: builder — install Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runner — minimal production image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runner

WORKDIR /app

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/best_model.joblib \
    MODEL_METADATA_PATH=/app/models/model_metadata.json \
    FEATURE_SCHEMA_PATH=/app/models/feature_schema.json \
    API_VERSION=0.1.0

# Create non-root user before any file ownership setup
RUN groupadd --system --gid 1001 appgroup \
    && useradd --system --uid 1001 --gid appgroup appuser

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY src/ ./src/

# Create the models mount-point directory owned by appuser.
# At runtime, mount the local models/ directory here:
#   -v "$(pwd)/models:/app/models:ro"
RUN mkdir -p /app/models && chown appuser:appgroup /app/models

USER appuser

EXPOSE 8000

# Health check uses Python stdlib (no extra packages required).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
