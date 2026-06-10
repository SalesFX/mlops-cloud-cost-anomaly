# ADR-001 — Modular Architecture

**Status:** Draft  
**Date:** 2026-06-10  
**Deciders:** Samuel Sales (architect)  
**Phase:** 0

---

## Context

The Cloud Cost Anomaly Detection Platform must evolve across six distinct phases: from a local ML script to a cloud-native, multi-cloud production system. Each phase adds new layers (API, observability, containers, orchestration, cloud connectors) without breaking existing ones.

Without deliberate architectural boundaries, early coupling decisions will make it painful to add new layers, replace components, or onboard contributors who only work on one area (e.g., a DevOps engineer who should not need to understand the ML pipeline).

---

## Decision

Adopt a **modular, layered architecture** from day one, where each major concern lives in its own directory with a well-defined interface.

The four primary modules are:

| Module | Path | Responsibility |
|--------|------|---------------|
| ML | `src/ml/` | Data generation, feature engineering, model training, prediction |
| API | `src/api/` | HTTP interface exposing ML predictions |
| Collectors | `src/collectors/` | Cloud billing data ingestion (AWS, OCI) |
| Observability | `src/observability/` | Metrics, structured logs, tracing |

Infrastructure artifacts (Docker, Helm, ArgoCD) live in `infra/` and are not imported by application code.

### Interface contracts

- The ML layer exposes a `predict(record: dict) -> dict` interface. No HTTP knowledge.
- The API layer imports from `src/ml/` only through that interface. It does not know about training.
- Collectors produce records conforming to the same schema as the synthetic dataset. The ML layer is agnostic to the source.
- Observability is injected via middleware or decorators — modules do not call Prometheus directly.

### Dependency rule

```
collectors → ml ← api
                   ↑
              observability (cross-cutting, injected)
```

No module imports from `infra/`. No module imports from `src/api/` within `src/ml/`.

---

## Consequences

**Positive:**
- Each module can be developed, tested, and reviewed independently.
- A new collector (e.g., GCP Billing) can be added without touching ML or API code.
- The ML layer can be replaced or upgraded without changing the API contract.
- Contributors can own a single module without understanding the full system.

**Negative:**
- Requires discipline to maintain boundaries — easy to accidentally import across layers.
- Initial boilerplate is higher (each module needs its own `__init__.py`, tests, etc.).

**Mitigations:**
- `CLAUDE.md` enforces the dependency rule explicitly.
- Code review checklist (via `reviewer` agent) includes a cross-import check.

---

## Alternatives Considered

### Monolithic single-file script
Rejected. Acceptable for a proof-of-concept, but creates high coupling that makes future phases expensive to implement.

### Microservices from day one
Rejected. Premature for a portfolio project in Phase 0. Each service adds networking, deployment, and observability overhead that is unjustified before Phase 5.

---

## References

- `docs/project-spec.md` — Section 4: Architecture Principles
- `ADR-004-layer-separation.md` — Detailed layer boundary decisions
