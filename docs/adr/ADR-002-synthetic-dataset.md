# ADR-002 — Uso de Dataset Sintético na Fase Demo

**Status:** Draft  
**Date:** 2026-06-10  
**Deciders:** Samuel Sales (architect)  
**Phase:** 0

---

## Context

The platform's ML pipeline requires billing cost data to train and validate anomaly detection models. Options include:

1. **Real billing data** — actual AWS/OCI cost records from a live account.
2. **Public benchmark datasets** — existing public cloud cost datasets.
3. **Synthetic generated data** — programmatically generated records that mimic real billing structure.

Real billing data cannot be used because:
- It exposes account IDs, resource names, and cost patterns that are sensitive.
- It requires active cloud accounts with billing enabled — an external dependency.
- It introduces compliance risk if committed to a version-controlled repository.
- It makes the project non-reproducible for others who want to run it locally.

Public benchmark datasets are rare for cloud billing and often lack the schema flexibility needed to demonstrate feature engineering choices.

---

## Decision

Use **programmatically generated synthetic billing data** (CSV format) for all ML development in Phase 1.

### Synthetic data requirements

The generator must produce records conforming to the canonical data model defined in `docs/project-spec.md` Section 6:

| Field | Type | Generation rule |
|-------|------|----------------|
| `date` | date | Sequential daily range |
| `service` | string | Sampled from a fixed list of realistic service names |
| `resource_id` | string | Synthetic identifier with no real account reference |
| `region` | string | Sampled from realistic region names |
| `usage_type` | string | Sampled per service |
| `cost_usd` | float | Normal distribution with service-specific mean and std |
| `is_anomaly` | bool | Injected at configurable rate (default 2–5%) |

### Anomaly injection strategy

Anomalies are injected as **cost multipliers** (e.g., 5×–20× the normal baseline for a given service) applied to randomly selected records. This ensures:

- Ground truth labels are available for supervised models.
- The injection rate is configurable to test models under different anomaly frequencies.
- Anomalies are statistically detectable but not trivially obvious.

### Isolation from real data

- `data/` directory is added to `.gitignore` (except `data/fixtures/` for small test fixtures).
- Generator script never reads from environment variables, AWS profiles, or OCI config files.
- No real region names are used in a way that implies actual account access.

---

## Consequences

**Positive:**
- Project is fully reproducible on any machine without cloud credentials.
- Data schema is fully controlled — easy to add new fields for feature engineering.
- Anomaly injection rate is configurable — supports model robustness testing.
- No compliance or privacy risk.

**Negative:**
- Synthetic data does not capture real-world billing quirks (partial usage charges, support fee tiers, discount programs).
- Model performance on synthetic data may not generalize to real billing data without retraining.

**Mitigations:**
- The collector interface (Phase 6, ADR-005) ensures the real data path produces the same schema as the synthetic generator. Retraining is isolated to the ML layer.
- The generator is designed to approximate realistic cost distributions — not random noise.

---

## Alternatives Considered

### Use a public cloud cost dataset
Rejected. No widely adopted public dataset matches the schema flexibility needed, and importing an external dataset creates a maintenance dependency.

### Use real billing data with anonymization
Rejected. Anonymization of billing data is non-trivial (resource IDs, service names, and cost patterns can be re-identified). The risk is not justified for a portfolio project.

---

## References

- `docs/project-spec.md` — Section 6: Data Model
- `ADR-005-cloud-billing-integration.md` — Strategy for replacing synthetic data with real billing in Phase 6
