# Project Specification — Cloud Cost Anomaly Detection Platform

**Version:** 0.1.0  
**Status:** Phase 0 — Organization  
**Last updated:** 2026-06-10

---

## 1. Problem Statement

Cloud infrastructure costs are dynamic and difficult to monitor manually. Unexpected cost spikes — caused by misconfigured resources, runaway workloads, or billing anomalies — often go undetected until the monthly invoice arrives.

This platform provides automated anomaly detection over cloud billing data, combining statistical baselines with machine learning models to surface cost anomalies in near real-time.

---

## 2. Goals

### Primary (Phase 1)
- Detect cost anomalies in billing data using ML models.
- Support unsupervised detection (no labeled anomalies required).
- Produce explainable predictions (not black-box only).
- Save and version the best-performing model.

### Secondary (Phase 2–5)
- Expose predictions via a REST API.
- Provide observability metrics and dashboards.
- Package the platform as a containerized service.
- Deploy on Kubernetes with production-grade configuration.

### Future (Phase 6)
- Ingest real billing data from AWS Cost Explorer, AWS CUR, and OCI Cost Reports.
- Support multi-cloud cost aggregation.

---

## 3. Non-Goals

- Real-time streaming ingestion (batch processing only in Phase 1).
- Multi-tenancy or user authentication (out of scope for portfolio).
- Cost optimization recommendations (detection only, not remediation).
- Financial forecasting or budgeting.

---

## 4. Architecture Principles

1. **Modular** — ML, API, collectors, and infra are independent layers with no circular dependencies.
2. **Replaceable** — The synthetic dataset can be swapped for real billing data without changing the ML layer.
3. **Explainable** — At least one model (Decision Tree) must produce human-readable feature importance.
4. **Observable** — Every layer emits structured logs and metrics.
5. **Testable** — Every module has accompanying unit tests before being considered complete.

---

## 5. Phases

### Phase 0 — Organization (current)
- CLAUDE.md, agents, ADRs, project spec.
- No functional code.
- **Exit criteria:** All governance files created and reviewed.

### Phase 1 — ML/FinOps Core
- Synthetic billing dataset generator (Python, Pandas).
- Feature engineering pipeline.
- Statistical baseline (z-score, IQR-based anomaly flags).
- Isolation Forest (unsupervised anomaly detection).
- Decision Tree classifier (supervised, explainable).
- XGBoost classifier (supervised, high performance).
- Model comparison: precision, recall, F1, ROC-AUC.
- Best model saved with Joblib + `metadata.json`.
- **Exit criteria:** All models trained, metrics compared, best model saved.

### Phase 2 — API
- FastAPI application.
- `POST /predict` — receive cost record, return anomaly prediction.
- `GET /health` — liveness check.
- `GET /model/info` — return model metadata.
- Pydantic v2 input/output validation.
- **Exit criteria:** All endpoints tested with unit and integration tests.

### Phase 3 — Observability (local)
- Prometheus metrics endpoint (`/metrics`).
- Grafana dashboard (prediction count, latency, anomaly rate).
- Structured JSON logs via `structlog`.
- **Exit criteria:** Metrics scraped by Prometheus, dashboard importable.

### Phase 4 — DevOps
- Dockerfile (multi-stage build).
- `docker-compose.yml` (app + Prometheus + Grafana).
- GitHub Actions workflow (lint, test, build, Trivy scan).
- **Exit criteria:** CI passes, Docker image builds and runs locally.

### Phase 5 — Cloud Native
- Kubernetes manifests (Deployment, Service, ConfigMap, HPA).
- Helm chart.
- ArgoCD Application manifest.
- Ingress configuration.
- **Exit criteria:** App deploys to local cluster (minikube/kind), HPA triggers on load.

### Phase 6 — Cloud Billing Integration
- `collectors/aws_cost_explorer.py` — AWS Cost Explorer connector.
- `collectors/aws_cur.py` — AWS CUR (S3) connector.
- `collectors/oci_cost_reports.py` — OCI Cost Reports connector.
- Collector interface with same schema as synthetic dataset.
- **Exit criteria:** Collector returns same schema as synthetic generator; ML layer unchanged.

---

## 6. Data Model (Phase 1)

Synthetic billing records follow this schema (implemented in `src/ml/generate_dataset.py`):

| Column | Type | Description |
|--------|------|-------------|
| `date` | string (ISO 8601) | Billing date |
| `provider` | string | Cloud provider: `AWS` or `OCI` |
| `account_id` | string | Synthetic account/tenancy ID |
| `service` | string | Cloud service name (EC2, S3, RDS, Compute, etc.) |
| `region` | string | Cloud region |
| `environment` | string | Workload environment: `dev`, `staging`, `prod` |
| `resource_id` | string | Stable synthetic resource identifier |
| `tag_project` | string | Cost allocation tag — project name |
| `tag_owner` | string | Cost allocation tag — owner name |
| `daily_cost` | float | Daily cost in USD |
| `usage_quantity` | float | Daily usage quantity |
| `currency` | string | Always `USD` |
| `is_anomaly` | bool | Ground truth label (for supervised models) |
| `anomaly_type` | string | `none`, `cost_spike`, `usage_spike`, `missing_tag`, `unexpected_service_growth` |

**Providers and services:**

| Provider | Services |
|----------|---------|
| AWS | EC2, RDS, S3, Lambda, EKS |
| OCI | Compute, Autonomous Database, Object Storage, OKE, Load Balancer |

See `ADR-002-synthetic-dataset.md` for the full schema rationale and anomaly injection strategy.

### Engineered features (Phase 1.2)

The feature engineering pipeline (`src/ml/feature_engineering.py`) adds 10 derived columns:

| Feature | Source columns | Notes |
|---------|---------------|-------|
| `previous_day_cost` | `daily_cost` | Per-resource rolling shift(1) |
| `previous_day_usage` | `usage_quantity` | Per-resource rolling shift(1) |
| `avg_cost_7d` | `daily_cost` | 7-day rolling mean, min_periods=1 |
| `avg_cost_30d` | `daily_cost` | 30-day rolling mean, min_periods=1 |
| `cost_change_percent` | `daily_cost`, `previous_day_cost` | NaN when no previous record |
| `usage_change_percent` | `usage_quantity`, `previous_day_usage` | NaN when no previous record |
| `cost_to_usage_ratio` | `daily_cost`, `usage_quantity` | NaN when usage = 0 |
| `is_missing_tag` | `tag_project`, `tag_owner` | True when either is empty |
| `day_of_week` | `date` | 0 = Monday, 6 = Sunday |
| `is_weekend` | `day_of_week` | True when day_of_week >= 5 |

**Rolling feature grouping — architecture note:**

- **Phase 1.2 (synthetic data):** group by `resource_id`. In the synthetic dataset, `account_id` and `region` are randomly assigned per day and not stable across days for the same resource. Grouping by them would produce groups of 1–3 records with meaningless rolling averages.
- **Phase 6 (real billing data):** recommended group is `provider + account_id + region + resource_id`. All four fields are stable per resource in real billing systems. This change is isolated to `add_rolling_features()`.

---

## 7. Model Strategy

See ADR-003 for full rationale.

| Model | Type | Purpose |
|-------|------|---------|
| Statistical baseline | Rule-based | Z-score and IQR flags; zero ML required |
| Isolation Forest | Unsupervised ML | Detect anomalies without labels |
| Decision Tree | Supervised ML | Explainable predictions, feature importance |
| XGBoost | Supervised ML | High-performance predictions |

---

## 8. Tech Stack

| Layer | Technology | Phase |
|-------|-----------|-------|
| Language | Python 3.11+ | 1+ |
| Data | Pandas | 1+ |
| ML | Scikit-learn, XGBoost, Joblib | 1+ |
| API | FastAPI, Pydantic v2, Uvicorn | 2+ |
| Observability | Prometheus client, Grafana, structlog | 3+ |
| Containerization | Docker, docker-compose | 4+ |
| Orchestration | Kubernetes, Helm, ArgoCD | 5+ |
| Cloud (AWS) | boto3, AWS Cost Explorer, S3 | 6+ |
| Cloud (OCI) | oci-sdk, OCI Cost Reports | 6+ |

---

## 9. Repository Layout

```
.
├── CLAUDE.md                   # Agent rules and phase gates
├── docs/
│   ├── project-spec.md         # This file
│   └── adr/                    # Architecture Decision Records
├── src/
│   ├── ml/                     # Phase 1
│   ├── api/                    # Phase 2
│   ├── collectors/             # Phase 6
│   └── observability/          # Phase 3
├── infra/                      # Phase 4–5
├── tests/                      # Phase 1+
├── data/                       # Phase 1 (gitignored)
└── models/                     # Phase 1 (gitignored)
```

---

## 10. Success Criteria (End of Phase 1)

- [ ] Synthetic dataset generated with configurable anomaly injection rate.
- [ ] Feature engineering pipeline produces consistent feature matrix.
- [ ] Isolation Forest detects injected anomalies with recall > 0.80.
- [ ] XGBoost achieves F1 > 0.85 on held-out test split.
- [ ] Decision Tree produces readable feature importance.
- [ ] Best model saved to `models/` with accompanying `metadata.json`.
- [ ] All modules have unit tests with coverage > 80%.
