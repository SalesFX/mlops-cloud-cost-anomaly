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

## Phase 1.2 — Feature Engineering

### What it does

`feature_engineering.py` reads the raw billing CSV and produces an enriched dataset with 10 new features ready for statistical baseline and ML training.

### Running

```bash
python src/ml/feature_engineering.py \
    --input data/cloud_costs.csv \
    --output data/cloud_cost_features.csv
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | `data/cloud_costs.csv` | Raw billing CSV (output of Phase 1.1) |
| `--output` | `data/cloud_cost_features.csv` | Enriched output CSV |

### New feature columns

| Feature | Description | Fallback |
|---------|-------------|---------|
| `previous_day_cost` | Cost of this resource on the previous day | `NaN` (first day) |
| `previous_day_usage` | Usage of this resource on the previous day | `NaN` (first day) |
| `avg_cost_7d` | 7-day rolling mean cost per resource | Uses available days (`min_periods=1`) |
| `avg_cost_30d` | 30-day rolling mean cost per resource | Uses available days (`min_periods=1`) |
| `cost_change_percent` | `(cost - prev_cost) / prev_cost * 100` | `NaN` when no previous record |
| `usage_change_percent` | `(usage - prev_usage) / prev_usage * 100` | `NaN` when no previous record |
| `cost_to_usage_ratio` | `daily_cost / usage_quantity` | `NaN` when usage is zero |
| `is_missing_tag` | `True` if `tag_project` or `tag_owner` is empty | — |
| `day_of_week` | Integer 0 (Mon) – 6 (Sun) | — |
| `is_weekend` | `True` if `day_of_week >= 5` | — |

### Rolling feature grouping — architecture note

Rolling features (`previous_day_cost`, `previous_day_usage`, `avg_cost_7d`, `avg_cost_30d`, `cost_change_percent`, `usage_change_percent`) are grouped by **`resource_id`** in Phase 1.2.

**Why `resource_id` only (Phase 1.2):**
In the synthetic dataset, `account_id` and `region` are assigned randomly per day and are not stable across days for the same resource. Grouping by them would produce groups of 1–3 records, making rolling averages meaningless.

**Recommended group in real billing data (Phase 6):**
When real collectors (AWS, OCI) are used, `account_id` and `region` will be stable for a given resource. The group should become:
```
provider + account_id + region + resource_id
```
This change is isolated to `add_rolling_features()` — no other function needs to change.

### Acceptance criteria (Phase 1.2)

- [x] `data/cloud_cost_features.csv` is generated from `data/cloud_costs.csv`
- [x] Output has the same number of rows as the input (no records dropped)
- [x] All 10 new feature columns are present
- [x] Rolling features are calculated per `resource_id` (groups never cross resource boundaries)
- [x] First record of each resource has `NaN` for `previous_day_cost` and `previous_day_usage`
- [x] `is_missing_tag` is `True` when `tag_project` or `tag_owner` is empty string
- [x] `cost_to_usage_ratio` is `NaN` when `usage_quantity == 0` (no division by zero)
- [x] `is_anomaly` and `anomaly_type` columns are preserved unchanged
- [x] Empty tag strings are loaded from CSV correctly (`keep_default_na=False`)
- [x] All tests in `tests/ml/test_feature_engineering.py` pass

### Running tests

```bash
pytest tests/ml/test_feature_engineering.py -v
# or all ML tests:
pytest tests/ -v
```

---

## Phase 1.3 — Statistical Baseline Detector

### What it does

`baseline_detector.py` applies five rule-based anomaly detection rules to the feature-enriched dataset, adding four `baseline_*` columns. Serves as the non-ML comparison baseline for Isolation Forest, Decision Tree and XGBoost.

### Running

```bash
python src/ml/baseline_detector.py \
    --input  data/cloud_cost_features.csv \
    --output data/cloud_cost_baseline_predictions.csv
```

### Rules and thresholds

| Rule | Condition | Reason label |
|------|-----------|-------------|
| 1 | `daily_cost > avg_cost_7d * 2.5` | `cost_above_7d_average` |
| 2 | `daily_cost > avg_cost_30d * 3.0` | `cost_above_30d_average` |
| 3 | `cost_change_percent >= 150` | `high_cost_change` |
| 4 | `usage_change_percent >= 200` | `high_usage_change` |
| 5 | `is_missing_tag == True` | `missing_tag` |

**Priority** (when multiple rules fire, the highest-priority reason wins):

```
none < missing_tag < high_usage_change < high_cost_change
     < cost_above_7d_average < cost_above_30d_average
```

`cost_above_30d_average` has the highest priority because a violation of the long-term baseline is a stronger anomaly signal than a short-term spike.

### Scoring

| reason | baseline_score | baseline_risk_level |
|--------|---------------|-------------------|
| `none` | 0.0 | `low` |
| `missing_tag` | 0.30 | `medium` |
| `high_usage_change` | 0.50 | `medium` |
| `high_cost_change` | 0.60 | `high` |
| `cost_above_7d_average` | 0.75 | `high` |
| `cost_above_30d_average` | 0.85 | `high` |

### Output columns added

| Column | Type | Description |
|--------|------|-------------|
| `baseline_anomaly` | bool | True when any rule fires |
| `baseline_score` | float [0–1] | Severity score of the primary reason |
| `baseline_risk_level` | string | `low` / `medium` / `high` |
| `baseline_reason` | string | Primary rule that triggered the flag |

### Acceptance criteria (Phase 1.3)

- [x] All rows from input preserved in output
- [x] Four `baseline_*` columns added correctly
- [x] `baseline_score` in [0.0, 1.0] for all records
- [x] `baseline_risk_level` is one of `low`, `medium`, `high`
- [x] `baseline_reason` is one of the six valid values
- [x] `cost_above_30d_average` beats `cost_above_7d_average` when both fire
- [x] `cost_above_7d_average` beats `high_cost_change` when both fire
- [x] `missing_tag` alone yields `medium` risk, not `high`
- [x] NaN in `cost_change_percent` / `usage_change_percent` does not trigger rules
- [x] `is_anomaly` and `anomaly_type` preserved unchanged
- [x] No ML model trained

### Running tests

```bash
pytest tests/ml/test_baseline_detector.py -v
```

---

## Phase 1.4 — Isolation Forest Detector

### What it does

`isolation_forest_detector.py` trains an `IsolationForest` (scikit-learn) on the feature-enriched dataset without using any labels. First real ML model in the pipeline — fully unsupervised.

### Running

```bash
python src/ml/isolation_forest_detector.py \
    --input  data/cloud_cost_features.csv \
    --output data/cloud_cost_iforest_predictions.csv \
    --contamination 0.05 \
    --seed 42
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | `data/cloud_cost_features.csv` | Feature-enriched CSV (Phase 1.2 output) |
| `--output` | `data/cloud_cost_iforest_predictions.csv` | Predictions output |
| `--contamination` | `0.05` | Expected fraction of anomalies |
| `--seed` | `42` | Random seed for reproducibility |

### Features used for training

```python
["daily_cost", "usage_quantity", "previous_day_cost", "previous_day_usage",
 "avg_cost_7d", "avg_cost_30d", "cost_change_percent", "usage_change_percent",
 "cost_to_usage_ratio", "is_missing_tag", "day_of_week", "is_weekend"]
```

`is_anomaly` and `anomaly_type` are **never** used as features — preserved in output for evaluation only.

**Note on `is_missing_tag`:** Included as an explicit FinOps governance signal. Records with empty cost-allocation tags form a distinct cluster in feature space, making `missing_tag` anomalies easier to isolate unsupervised.

### Output columns added

| Column | Type | Description |
|--------|------|-------------|
| `iforest_anomaly` | bool | True when IsolationForest flags the record |
| `iforest_score` | float [0–1] | 0 = normal, 1 = most anomalous |
| `iforest_risk_level` | string | `low` (`< 0.4`) / `medium` (`0.4–0.7`) / `high` (`≥ 0.7`) |

### Evaluation note

`evaluate()` compares `iforest_anomaly` against `is_anomaly` ground truth. These metrics are a **benchmark for the synthetic scenario only** — not a proxy for production performance. In Phase 6, real data has no labels.

### Model persistence

No model artefact saved (`.pkl` / `.joblib`). Deferred to the model registry phase.

### Acceptance criteria (Phase 1.4)

- [x] Output has same row count as input
- [x] Three `iforest_*` columns added correctly
- [x] `iforest_score` in [0.0, 1.0] for all records
- [x] `is_anomaly` and `anomaly_type` preserved, never used as features
- [x] No model saved to disk
- [x] Reproducible with same `--seed`
- [x] All 35 tests in `tests/ml/test_isolation_forest_detector.py` pass

### Running tests

```bash
pytest tests/ml/test_isolation_forest_detector.py -v
```

---

## Upcoming (Phase 1.5+)

- `decision_tree_detector.py` — supervised, explainable classifier
- `xgboost_detector.py` — supervised, high-performance classifier
- `evaluator.py` — shared model comparison metrics
- `predictor.py` — unified `predict()` interface
- `model_registry.py` — model save/load with `metadata.json`
- `evaluator.py` — shared model comparison metrics
- `predictor.py` — unified `predict()` interface
- `model_registry.py` — model save/load with `metadata.json`
