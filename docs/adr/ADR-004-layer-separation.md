# ADR-004 — Separação entre ML, API, Collectors e Infraestrutura

**Status:** Draft  
**Date:** 2026-06-10  
**Deciders:** Samuel Sales (architect)  
**Phase:** 0

---

## Context

As the platform evolves across six phases, four major technical concerns must be developed in parallel by different contributors (or different agent personas):

1. **ML pipeline** — data ingestion, feature engineering, model training, prediction.
2. **API layer** — HTTP interface for predictions, model metadata, health checks.
3. **Collectors** — cloud billing data connectors for AWS and OCI.
4. **Infrastructure** — Docker, Kubernetes, Helm, ArgoCD, CI/CD.

Without explicit separation, these concerns tend to bleed into each other:
- ML engineers import FastAPI objects directly into training scripts.
- API routes call cloud SDK methods directly.
- Infrastructure configuration embeds application secrets.
- Adding a new cloud connector requires touching the ML code.

This coupling makes independent development and testing impossible.

---

## Decision

Each concern is separated into its own directory with a **strict dependency rule**:

```
src/
├── ml/           # Core ML logic — no HTTP, no cloud SDK, no Prometheus
├── api/          # HTTP layer — imports from src/ml/ only via prediction interface
├── collectors/   # Cloud billing connectors — imports from src/ml/ only to produce records
└── observability/# Cross-cutting metrics/logging — injected, not imported directly
infra/            # Infrastructure only — no Python imports
```

### Dependency rules

| Module | May import | Must NOT import |
|--------|-----------|----------------|
| `src/ml/` | stdlib, pandas, scikit-learn, xgboost, joblib | fastapi, boto3, oci, prometheus_client |
| `src/api/` | fastapi, pydantic, `src/ml/` | boto3, oci, pandas, sklearn |
| `src/collectors/` | boto3, oci, pandas | fastapi, sklearn, xgboost |
| `src/observability/` | prometheus_client, structlog | fastapi, boto3, sklearn |
| `infra/` | (not Python — YAML/HCL only) | — |

### Interface boundaries

**ML → API contract:**  
`src/ml/predictor.py` exposes a single function:
```python
def predict(record: dict) -> dict:
    # returns {"is_anomaly": bool, "anomaly_score": float, "model": str}
```
The API layer calls this function. It does not import model files directly.

**Collector → ML contract:**  
Each collector produces a list of dicts conforming to the canonical billing schema (see `docs/project-spec.md` Section 6). The ML layer is agnostic to whether the source is synthetic or real.

**Observability contract:**  
Observability is injected via a decorator or middleware pattern — modules do not call `prometheus_client.Counter.inc()` directly in business logic.

---

## Internal module structure (Phase 1 reference)

```
src/ml/
├── __init__.py
├── data_generator.py      # Synthetic data generation
├── feature_engineering.py # Feature transformations
├── baseline.py            # Statistical z-score/IQR detector
├── isolation_forest.py    # Isolation Forest wrapper
├── decision_tree.py       # Decision Tree wrapper
├── xgboost_model.py       # XGBoost wrapper
├── evaluator.py           # Shared metrics comparison
├── predictor.py           # Unified predict() interface
└── model_registry.py      # Save/load model + metadata.json
```

This internal structure is defined here as a reference for Phase 1 planning. It may be adjusted during Phase 1 with a corresponding ADR update.

---

## Consequences

**Positive:**
- ML, API, and collector modules can be developed and tested independently.
- A new cloud connector (e.g., GCP) only requires a new file in `src/collectors/`.
- The API can be replaced (e.g., gRPC instead of HTTP) without touching the ML layer.
- Infrastructure never imports application code — no circular dependency risk.

**Negative:**
- Requires explicit interface definitions between layers before implementation.
- Contributors must understand the boundary rules (enforced via `CLAUDE.md`).

**Mitigations:**
- `CLAUDE.md` documents the dependency rule explicitly.
- The `reviewer` agent checks for cross-layer imports in every code review.
- The `predictor.py` interface is defined and frozen before the API layer begins.

---

## Alternatives Considered

### Feature-based organization (e.g., `src/anomaly_detection/`, `src/billing/`)
Rejected. Feature-based organization works well for frontend applications but leads to ML/HTTP coupling in backend ML systems. Layer-based separation matches the team structure (ML engineers, backend engineers, DevOps).

### Single `src/` flat structure
Rejected. Flat structure is acceptable for scripts but makes enforcing boundaries impossible at scale.

---

## References

- `docs/project-spec.md` — Section 4: Architecture Principles, Section 9: Repository Layout
- `ADR-001-modular-architecture.md` — High-level modularity decision
- `ADR-005-cloud-billing-integration.md` — Collector layer strategy
