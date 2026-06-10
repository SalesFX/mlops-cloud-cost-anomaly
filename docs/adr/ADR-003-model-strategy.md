# ADR-003 — Estratégia de Modelos: Baseline Estatístico, Isolation Forest, Decision Tree e XGBoost

**Status:** Draft  
**Date:** 2026-06-10  
**Deciders:** Samuel Sales (architect, ml-engineer)  
**Phase:** 0

---

## Context

Anomaly detection in cloud billing data presents a mixed problem:

- **Unsupervised** — In real production environments, labeled anomalies rarely exist. The system must detect anomalies without ground truth.
- **Supervised** — In Phase 1, we inject known anomalies into synthetic data, so ground-truth labels are available for supervised training and evaluation.
- **Explainability** — FinOps stakeholders need to understand why a cost record was flagged as anomalous. Black-box models alone are insufficient.
- **Performance** — The platform must scale to large billing datasets. The model must be fast at inference time.

A single model type cannot satisfy all four requirements simultaneously. A multi-model strategy covering different detection paradigms is needed.

---

## Decision

Implement and compare **four detection approaches** in Phase 1:

### 1. Statistical Baseline

**Type:** Rule-based, no ML  
**Method:** Z-score and IQR (interquartile range) thresholds computed per `(service, region)` group.

- A record is flagged if its `cost_usd` deviates more than N standard deviations from the group mean (z-score threshold, configurable, default N=3).
- Alternatively flagged if it falls outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`.
- Provides a **non-ML baseline** to compare against. Anything below this baseline in precision/recall is a failure.

**Why include it:**  
Demonstrates that simple statistics often outperform complex models on clean, structured data. Forces honest model evaluation.

---

### 2. Isolation Forest

**Type:** Unsupervised ML (Scikit-learn)  
**Use case:** Detect anomalies without labels — the realistic production scenario.

- Isolates anomalies by randomly partitioning the feature space. Anomalies are isolated in fewer partitions.
- Produces an `anomaly_score` (continuous) in addition to a binary prediction.
- Trained on the full dataset (including injected anomalies, but without seeing the label).

**Why include it:**  
Represents the real-world deployment scenario where labeled data does not exist. Demonstrates unsupervised ML capability.

---

### 3. Decision Tree Classifier

**Type:** Supervised ML (Scikit-learn)  
**Use case:** Explainable anomaly classification using ground-truth labels.

- Trained on labeled synthetic data (`is_anomaly` column).
- Produces a human-readable decision tree with feature importance scores.
- Depth is intentionally constrained (max_depth configurable, default 5) to maintain interpretability.

**Why include it:**  
Provides the **explainability story** — a non-technical stakeholder can follow the decision path. Also serves as a strong sanity check: if Decision Tree underperforms XGBoost significantly, it reveals which features are non-linear.

---

### 4. XGBoost Classifier

**Type:** Supervised ML (XGBoost)  
**Use case:** High-performance anomaly classification using ground-truth labels.

- Gradient boosted tree ensemble trained on labeled synthetic data.
- Produces probability scores, not just binary predictions.
- Feature importance available via `plot_importance` for post-hoc explainability.

**Why include it:**  
Represents the production-grade supervised model. Expected to achieve the highest F1 and ROC-AUC. Demonstrates MLOps maturity (hyperparameter logging, model versioning).

---

## Model Comparison Protocol

All four models are evaluated on the same held-out test split (80/20 train/test, stratified by `is_anomaly`).

Metrics collected for each model:

| Metric | Why |
|--------|-----|
| Precision | Avoid false positives — unnecessary alerts |
| Recall | Avoid false negatives — missed anomalies |
| F1 Score | Harmonic mean, primary ranking metric |
| ROC-AUC | Ranking quality independent of threshold |
| Inference latency (ms) | Runtime constraint |

The model with the **highest F1 on the test split** is saved as the production model. In case of a tie, XGBoost is preferred over Decision Tree; Decision Tree over Isolation Forest.

---

## Model Persistence

The saved model includes:
- `models/best_model.joblib` — serialized model artifact.
- `models/metadata.json` — model name, training date, F1, precision, recall, ROC-AUC, feature list, threshold used.

---

## Consequences

**Positive:**
- Multi-paradigm coverage: unsupervised + supervised + rule-based.
- Honest baseline comparison — no "best model" claim without proof.
- Explainability story via Decision Tree.
- Feature importance available for both Decision Tree and XGBoost.

**Negative:**
- Four models increase Phase 1 implementation scope.
- Comparison adds evaluation infrastructure (metrics table, report generation).

**Mitigations:**
- Scikit-learn's unified `fit`/`predict` interface minimizes per-model boilerplate.
- A shared evaluation function handles all four models.

---

## Alternatives Considered

### XGBoost only
Rejected. Misses the unsupervised scenario and provides no explainability story.

### Deep learning (LSTM for time-series anomaly detection)
Deferred. Valid for future phases, but requires more data, more compute, and adds complexity without demonstrating additional MLOps concepts. Can be added as a fifth model in a future iteration.

### DBSCAN (density-based clustering)
Deferred. Interesting for spatial anomaly detection but adds tuning complexity (epsilon, min_samples) without clear advantage over Isolation Forest on tabular billing data.

---

## References

- `docs/project-spec.md` — Section 7: Model Strategy
- `ADR-002-synthetic-dataset.md` — Synthetic data with injected anomalies
