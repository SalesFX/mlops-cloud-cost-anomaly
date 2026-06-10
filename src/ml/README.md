# src/ml — ML Layer

This directory contains the ML pipeline for the Cloud Cost Anomaly Detection Platform.

## Phase 1.1 — Synthetic Dataset Generator

### What it does

`generate_dataset.py` generates a reproducible synthetic cloud billing dataset with controlled anomaly injection. The dataset represents daily cost records across AWS and OCI services, environments, and regions.

### Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Generating the dataset

```bash
python src/ml/generate_dataset.py --days 180 --output data/cloud_costs.csv --seed 42 --anomaly-rate 0.05
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--days` | 180 | Number of days of billing history |
| `--output` | `data/cloud_costs.csv` | Output CSV file path |
| `--seed` | 42 | Random seed for full reproducibility |
| `--anomaly-rate` | 0.05 | Fraction of anomalous records (0.0 to 1.0) |

### Dataset schema

| Column | Type | Description |
|--------|------|-------------|
| `date` | string (ISO 8601) | Billing date |
| `provider` | string | Cloud provider: `AWS` or `OCI` |
| `account_id` | string | Synthetic account/tenancy ID |
| `service` | string | Cloud service name |
| `region` | string | Cloud region |
| `environment` | string | Workload environment: `dev`, `staging`, `prod` |
| `resource_id` | string | Synthetic resource identifier |
| `tag_project` | string | Cost allocation tag — project name (empty if `missing_tag` anomaly) |
| `tag_owner` | string | Cost allocation tag — owner name (empty if `missing_tag` anomaly) |
| `daily_cost` | float | Daily cost in USD |
| `usage_quantity` | float | Daily usage quantity |
| `currency` | string | Always `USD` |
| `is_anomaly` | bool | Ground truth anomaly label |
| `anomaly_type` | string | `none`, `cost_spike`, `usage_spike`, `missing_tag`, `unexpected_service_growth` |

### Providers and services

| Provider | Services |
|----------|---------|
| AWS | EC2, RDS, S3, Lambda, EKS |
| OCI | Compute, Autonomous Database, Object Storage, OKE, Load Balancer |

### Anomaly types

| Type | Description |
|------|-------------|
| `cost_spike` | `daily_cost` multiplied by 5–20x |
| `usage_spike` | `usage_quantity` multiplied by 5–20x, cost increases proportionally |
| `missing_tag` | `tag_project` and `tag_owner` are empty strings |
| `unexpected_service_growth` | Both cost and usage multiplied by 2–5x |

### Environment cost ordering

`prod` costs ≈ 4.5× `dev`; `staging` ≈ 1.8× `dev`.

### Running tests

```bash
pytest tests/ml/test_generate_dataset.py -v
```

---

## Upcoming (Phase 1.2+)

- `feature_engineering.py` — feature transformations for ML models
- `baseline.py` — statistical z-score/IQR anomaly detection
- `isolation_forest.py` — unsupervised anomaly detection
- `decision_tree.py` — explainable supervised classifier
- `xgboost_model.py` — high-performance supervised classifier
- `evaluator.py` — shared model comparison metrics
- `predictor.py` — unified `predict()` interface
- `model_registry.py` — model save/load with `metadata.json`
