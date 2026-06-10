# CLAUDE.md — Cloud Cost Anomaly Detection Platform

## Project Overview

FinOps + MLOps platform for detecting cloud cost anomalies.  
Demo phase uses synthetic CSV data. Future phases will integrate with AWS Cost Explorer, AWS CUR, and OCI Cost Reports.

This project is a portfolio piece demonstrating DevOps, Cloud, and MLOps competencies.

---

## Current Phase

**Phase 0 — Organization**  
No functional code exists yet. Only governance, architecture decisions, and project structure.

---

## Phase Gates

Each phase must be explicitly approved before the next begins.

| Phase | Scope | Unlocks |
|-------|-------|---------|
| 0 | CLAUDE.md, ADRs, agents, project spec, directory structure | Phase 1 |
| 1 | Synthetic dataset, feature engineering, models, metrics | Phase 2 |
| 2 | FastAPI, endpoints, Pydantic validation | Phase 3 |
| 3 | Prometheus metrics, Grafana dashboard, structured logs | Phase 4 |
| 4 | Docker, docker-compose, tests, GitHub Actions, Trivy | Phase 5 |
| 5 | Kubernetes, Helm, ArgoCD, HPA, Ingress | Phase 6 |
| 6 | AWS Cost Explorer, AWS CUR, OCI Cost Reports, real collectors | — |

---

## Hard Constraints (Never Violate)

- **Never use real cloud credentials, account IDs, tenancy IDs, or tokens.**
- **Never connect to AWS, OCI, Azure, or any real cloud API.**
- **Never commit real billing data or cost records.**
- **Never skip an ADR for a relevant architectural decision.**
- **Never implement a phase before the previous phase is approved.**

---

## Per-Phase Restrictions

### Phase 0 (current)
- No Python code of any kind.
- No `requirements.txt`, `pyproject.toml`, or `setup.py`.
- No Dockerfile or docker-compose.
- No CI/CD pipelines.
- No Kubernetes, Helm, ArgoCD, or Terraform.
- No data files.
- No API.

### Phase 1
- No FastAPI or HTTP servers.
- No Docker or containerization.
- No real billing data — synthetic CSV only.
- No cloud SDK calls (boto3, oci-sdk, etc.).

### Phase 2
- No Docker yet.
- No Kubernetes yet.
- No real cloud integrations.

---

## Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Data processing | Pandas |
| ML models | Scikit-learn, XGBoost, Joblib |
| API (Phase 2+) | FastAPI, Pydantic, Uvicorn |
| Observability (Phase 3+) | Prometheus, Grafana, structlog |
| Containerization (Phase 4+) | Docker, docker-compose |
| Orchestration (Phase 5+) | Kubernetes, Helm, ArgoCD |
| Cloud integration (Phase 6+) | boto3 (AWS), oci-sdk (OCI) |

---

## Directory Structure

```
.
├── CLAUDE.md
├── docs/
│   ├── project-spec.md
│   └── adr/
│       ├── ADR-001-modular-architecture.md
│       ├── ADR-002-synthetic-dataset.md
│       ├── ADR-003-model-strategy.md
│       ├── ADR-004-layer-separation.md
│       └── ADR-005-cloud-billing-integration.md
├── src/
│   ├── ml/              # Phase 1: feature engineering, models, training pipeline
│   ├── api/             # Phase 2: FastAPI application
│   ├── collectors/      # Phase 6: AWS/OCI billing connectors
│   └── observability/   # Phase 3: Prometheus metrics, structured logs
├── infra/               # Phase 4-5: Docker, Helm, ArgoCD manifests
├── tests/               # All phases: unit and integration tests
├── data/                # Phase 1: synthetic CSVs (gitignored except fixtures)
└── models/              # Phase 1: saved model artifacts (gitignored)
```

---

## ADR Obligation

Every relevant architectural decision must be documented as an ADR in `docs/adr/`.

An ADR is required when:
- Choosing between two or more design approaches.
- Adding a new dependency to the project.
- Deciding how a module boundary will work.
- Making a trade-off between simplicity and future extensibility.

ADR template lives in `docs/adr/` — use the same header format as existing ADRs.

---

## Naming Conventions

- Files: `kebab-case.py`, `kebab-case.md`
- Python modules: `snake_case`
- Python classes: `PascalCase`
- Python functions and variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- ADRs: `ADR-NNN-short-title.md`

---

## Agent Roles

Each specialized agent lives in `.claude/agents/`. Invoke the appropriate agent for each task type.

| Agent | Responsibility |
|-------|---------------|
| `architect` | Architecture decisions, ADRs, layer design |
| `ml-engineer` | Models, features, metrics, training pipelines |
| `backend-engineer` | FastAPI, Pydantic, HTTP endpoints |
| `devops-engineer` | Docker, CI/CD, Kubernetes, Helm |
| `finops-engineer` | Billing connectors, cost data modeling, FinOps practices |
| `reviewer` | Cross-cutting code review, quality gates |

---

## Implementation Rules

- Every implementation must be small, focused, and independently reviewable.
- Tests must accompany every implemented module (Phase 1+).
- No half-finished code — a feature is either complete or not started.
- No placeholders, TODOs, or stubs in production paths.
- No premature abstraction — solve the current phase's problem only.
- Commits must be atomic and descriptive.
