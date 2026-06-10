# ADR-005 — Estratégia Futura de Integração com AWS/OCI Billing

**Status:** Draft  
**Date:** 2026-06-10  
**Deciders:** Samuel Sales (architect, finops-engineer)  
**Phase:** 0 (implementation deferred to Phase 6)

---

## Context

The platform's Phase 1 ML pipeline is built on synthetic billing data. In Phase 6, it must ingest real billing records from at least two cloud providers:

- **AWS Cost Explorer** — daily/hourly cost breakdown via REST API, no S3 required.
- **AWS Cost and Usage Report (CUR)** — detailed billing CSV/Parquet files exported to S3.
- **OCI Cost Reports** — monthly/daily CSV reports stored in an OCI Object Storage bucket.

The integration must be designed so that:

1. The ML layer does not change when switching from synthetic to real data.
2. Real credentials are never committed to version control.
3. A new cloud provider can be added without touching existing collectors.
4. The platform can run in demo mode (synthetic) and production mode (real) without code changes.

This ADR documents the architectural decision for that integration, even though implementation is deferred to Phase 6.

---

## Decision

Implement a **collector abstraction layer** in `src/collectors/` with a common interface. Each cloud provider is a separate module implementing that interface.

### Collector interface (to be implemented in Phase 6)

```python
# src/collectors/base.py
from abc import ABC, abstractmethod
from typing import Iterator

class BillingCollector(ABC):
    @abstractmethod
    def fetch_records(
        self,
        start_date: str,  # ISO 8601: "2025-01-01"
        end_date: str,    # ISO 8601: "2025-01-31"
    ) -> Iterator[dict]:
        """Yield billing records conforming to the canonical schema."""
        ...
```

Each collector yields dicts matching the schema in `docs/project-spec.md` Section 6. The ML layer consumes iterators of dicts — it does not know or care which collector produced them.

### Planned collectors

| Collector | File | Data source |
|-----------|------|------------|
| AWS Cost Explorer | `src/collectors/aws_cost_explorer.py` | AWS Cost Explorer API (GetCostAndUsage) |
| AWS CUR | `src/collectors/aws_cur.py` | S3 bucket with CUR files (Parquet/CSV) |
| OCI Cost Reports | `src/collectors/oci_cost_reports.py` | OCI Object Storage bucket |
| Synthetic (demo) | `src/collectors/synthetic.py` | Local generator — wraps `src/ml/data_generator.py` |

The `SyntheticCollector` is the only collector available in Phases 1–5. It wraps the existing data generator to conform to the same interface.

### Credential management strategy

- Credentials are **never** stored in code, config files, or environment files committed to the repository.
- AWS credentials: injected via AWS profiles, IAM roles, or environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`). Managed externally (AWS Secrets Manager, GitHub Actions OIDC).
- OCI credentials: injected via OCI config file or instance principal. Never hardcoded.
- In demo mode, no credential is required — `SyntheticCollector` is used.

### Mode switching

Mode is controlled by a single environment variable:

```
BILLING_SOURCE=synthetic   # default, Phase 1–5
BILLING_SOURCE=aws_ce      # Phase 6, AWS Cost Explorer
BILLING_SOURCE=aws_cur     # Phase 6, AWS CUR
BILLING_SOURCE=oci         # Phase 6, OCI Cost Reports
```

A factory function in `src/collectors/factory.py` resolves the appropriate collector at runtime. The ML pipeline does not contain any conditional logic for data sources.

---

## Schema Mapping

Real billing APIs use different field names than the canonical schema. Each collector is responsible for mapping its source fields to the canonical schema before yielding records.

Example for AWS Cost Explorer:

| AWS Cost Explorer field | Canonical field |
|------------------------|----------------|
| `TimePeriod.Start` | `date` |
| `Keys[0]` (service) | `service` |
| — (not available) | `resource_id` (generated) |
| `Keys[1]` (region) | `region` |
| `MetricResults[0].Amount` | `cost_usd` |
| — | `is_anomaly` (None for real data) |

For supervised models, `is_anomaly` is `None` for real billing records (no ground truth available). The platform defaults to Isolation Forest predictions when labels are absent.

---

## Consequences

**Positive:**
- ML layer is completely isolated from cloud SDK changes.
- Adding a new cloud provider requires only a new file in `src/collectors/`.
- Demo mode and production mode share the same inference code path.
- Credential management is explicit and auditable.

**Negative:**
- Schema mapping for real APIs is complex — CUR files have 100+ columns that must be reduced to the canonical schema.
- AWS CUR requires S3 access and potentially Athena or direct Parquet parsing — adds infrastructure dependency in Phase 6.
- OCI Cost Reports lag by 24–48 hours — not suitable for real-time alerting.

**Mitigations:**
- Schema mapping is isolated inside each collector — the ML layer never sees unmapped fields.
- CUR parsing complexity is contained in `aws_cur.py` using `pyarrow` or `pandas`.
- OCI latency is a known limitation documented in collector metadata.

---

## Deferred Decisions (Phase 6)

The following decisions are intentionally deferred to Phase 6 when real integration context is available:

- Whether to use AWS Cost Explorer API or CUR as the primary AWS source.
- Whether to store normalized billing records in a local database for faster re-training.
- Whether to implement incremental ingestion (only fetch records since last run).
- Multi-account support (AWS Organizations, OCI compartments).

---

## Alternatives Considered

### Direct boto3/oci-sdk calls in the ML pipeline
Rejected. Violates the layer separation defined in ADR-004. Makes unit testing the ML pipeline impossible without mocking cloud SDKs.

### ETL pipeline (Airflow, Prefect) for data ingestion
Deferred. Valid for production-scale multi-cloud ingestion, but adds significant infrastructure complexity before Phase 5. Can be adopted in Phase 6 if needed.

### Single unified billing schema with ORM (SQLAlchemy)
Deferred. Adds a database dependency before the platform has a use case that requires persistence. Evaluate in Phase 6.

---

## References

- `docs/project-spec.md` — Phase 6 definition
- `ADR-001-modular-architecture.md` — Collector as an independent module
- `ADR-002-synthetic-dataset.md` — SyntheticCollector wraps the demo generator
- `ADR-004-layer-separation.md` — Collector → ML contract
