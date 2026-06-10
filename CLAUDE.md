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

## Engineering Practices

These practices are mandatory for all implementation work from Phase 1 onwards.

### Test-Driven Development

- Write tests before or alongside implementation — never after.
- A module is not considered complete until its tests pass.
- Tests must cover: happy path, edge cases, and error/validation cases.
- Use `pytest`. No `unittest`-style test classes unless there is a clear organisational reason.

### Function and module design

- Keep functions small with a single, clear responsibility.
- If a function needs a comment to explain what it does, it should be split or renamed.
- Prefer flat module structures over deep class hierarchies.
- Apply OOP only when there is a concrete benefit — e.g. a shared interface across multiple implementations (collectors in Phase 6). Never apply it preemptively.
- Avoid overengineering: three similar functions are better than a premature abstraction.

### Type hints

- Use type hints on all function signatures where the types are non-obvious or where the function is part of a public interface.
- Do not annotate trivial single-line helpers where the type is self-evident.
- Prefer built-in types (`list`, `dict`, `str`) over `typing` aliases for Python 3.9+.

### Responsibility separation

The following responsibilities must remain in separate functions or modules — never merged into one:

| Responsibility | Phase 1 location |
|---------------|-----------------|
| Data generation | `src/ml/generate_dataset.py` |
| Anomaly injection | `generate_dataset.py` — `inject_anomalies()` |
| Feature engineering | `src/ml/feature_engineering.py` |
| Model training | Per-model files (`baseline.py`, `isolation_forest.py`, etc.) |
| Model evaluation | `src/ml/evaluator.py` |
| Model persistence | `src/ml/model_registry.py` |
| Prediction serving | `src/ml/predictor.py` (Phase 1) / `src/api/` (Phase 2) |

### Code quality

- Code must be readable without comments — well-named identifiers are preferred over inline explanations.
- Only add a comment when the **why** is non-obvious: a hidden constraint, a statistical choice, a workaround.
- No `print()` statements in library code — use return values and let the CLI layer handle output.
- No dead code, commented-out blocks, or unused imports in committed files.

### Documentation

- Every `src/` subdirectory must have a `README.md` documenting: what it does, how to run it, and its acceptance criteria for the current phase.
- Acceptance criteria must be explicit and checkable — not vague descriptions.
- Update the relevant `README.md` as part of the same commit that implements the feature.

### Governance compliance

- Every implementation must comply with `CLAUDE.md` phase gates, relevant ADRs, and `docs/project-spec.md`.
- If an implementation requires deviating from an ADR, update the ADR first — do not implement first and document later.
- After each sub-phase is implemented, invoke the `reviewer` agent to assess scope compliance, test quality, code quality, and adherence to architectural decisions before marking the phase complete.

---

## Implementation Rules

- Every implementation must be small, focused, and independently reviewable.
- Tests must accompany every implemented module (Phase 1+).
- No half-finished code — a feature is either complete or not started.
- No placeholders, TODOs, or stubs in production paths.
- No premature abstraction — solve the current phase's problem only.
- Commits must be atomic and descriptive.
