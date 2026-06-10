# ADR-002 — Uso de Dataset Sintético na Fase Demo

**Status:** Accepted  
**Date:** 2026-06-10  
**Updated:** 2026-06-10 (Phase 1.1 — schema expanded to match FinOps requirements)  
**Deciders:** Samuel Sales (architect)  
**Phase:** 1

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

Implementation: `src/ml/generate_dataset.py`

### Canonical data schema

| Column | Type | Generation rule |
|--------|------|----------------|
| `date` | string (ISO 8601) | Sequential daily range from `--days` to today |
| `provider` | string | Sampled from `{AWS, OCI}` |
| `account_id` | string | Synthetic identifier, no real account reference |
| `service` | string | Sampled per provider (5 services each) |
| `region` | string | Sampled from provider-specific region list |
| `environment` | string | Sampled from `{dev, staging, prod}` |
| `resource_id` | string | Stable synthetic ID, consistent across days |
| `tag_project` | string | Sampled from project catalog; empty for `missing_tag` anomalies |
| `tag_owner` | string | Sampled from owner catalog; empty for `missing_tag` anomalies |
| `daily_cost` | float | Log-normal distribution, service+environment baseline |
| `usage_quantity` | float | Log-normal distribution, service+environment baseline |
| `currency` | string | Always `USD` |
| `is_anomaly` | bool | Injected at configurable rate (default 5%) |
| `anomaly_type` | string | `none`, `cost_spike`, `usage_spike`, `missing_tag`, `unexpected_service_growth` |

### Schema note

The original draft of this ADR used a minimal schema (`date`, `service`, `resource_id`, `region`, `usage_type`, `cost_usd`, `is_anomaly`). The schema was expanded during Phase 1.1 implementation to reflect richer FinOps requirements:

- `provider` and `account_id` — required for multi-cloud aggregation (Phase 6).
- `environment` — critical for FinOps cost allocation; prod/staging/dev have distinct cost profiles.
- `tag_project` and `tag_owner` — represent cloud governance tags; their absence is a real anomaly type.
- `daily_cost` (renamed from `cost_usd`) — clearer name.
- `usage_quantity` — enables detection of usage-driven vs cost-driven anomalies.
- `currency` — foundation for multi-currency support in Phase 6.
- `anomaly_type` — enables supervised multi-class training and explainability.

### Providers and services

| Provider | Services |
|----------|---------|
| AWS | EC2, RDS, S3, Lambda, EKS |
| OCI | Compute, Autonomous Database, Object Storage, OKE, Load Balancer |

### Environment cost model

```
prod    ≈ 4.5× dev baseline
staging ≈ 1.8× dev baseline
```

### Anomaly injection strategy

Anomalies are injected post-generation by selecting random records and applying a transformation based on anomaly type:

| Anomaly type | Transformation |
|-------------|---------------|
| `cost_spike` | `daily_cost` × 5–20× |
| `usage_spike` | `usage_quantity` × 5–20×, `daily_cost` × usage_mult × 0.9 |
| `missing_tag` | `tag_project` and `tag_owner` set to empty string |
| `unexpected_service_growth` | Both `daily_cost` and `usage_quantity` × 2–5× |

This ensures:
- Ground truth labels are available for supervised models (`is_anomaly`, `anomaly_type`).
- The injection rate is configurable to test models under different anomaly frequencies.
- Anomalies are statistically detectable but not trivially obvious.
- Four distinct anomaly types cover cost, usage, governance, and growth patterns.

### Reproducibility

- Controlled by `--seed` parameter via `np.random.default_rng(seed)`.
- Same seed + same parameters = byte-for-byte identical CSV.
- Resource IDs are pre-generated once and reused across days (stable resource catalog).

### Isolation from real data

- `data/*.csv` is in `.gitignore` — generated CSVs are never committed.
- Generator never reads from environment variables, AWS profiles, or OCI config files.
- No real account IDs, tenancy IDs, or credentials of any kind.

---

## Consequences

**Positive:**
- Project is fully reproducible on any machine without cloud credentials.
- Rich schema supports multi-cloud, environment-based, and tag-based feature engineering.
- Four anomaly types enable multi-class supervised training.
- Anomaly injection rate is configurable.

**Negative:**
- Synthetic data does not capture real-world billing quirks (partial usage charges, support fee tiers, discount programs, negotiated pricing).
- Model performance on synthetic data may not generalize to real billing data without retraining.

**Mitigations:**
- The collector interface (Phase 6, ADR-005) ensures the real data path produces the same schema as the synthetic generator. Retraining is isolated to the ML layer.
- Generator uses log-normal distributions (realistic for cost data) rather than uniform random noise.

---

## Alternatives Considered

### Use a public cloud cost dataset
Rejected. No widely adopted public dataset matches the schema flexibility needed.

### Use real billing data with anonymization
Rejected. Anonymization of billing data is non-trivial. The risk is not justified for a portfolio project.

### Separate generator per provider (AWS-only, OCI-only scripts)
Rejected. A unified generator with provider-specific catalogs is simpler and ensures schema consistency.

---

## References

- `src/ml/generate_dataset.py` — Implementation
- `docs/project-spec.md` — Section 6: Data Model
- `ADR-005-cloud-billing-integration.md` — Strategy for replacing synthetic data with real billing in Phase 6
