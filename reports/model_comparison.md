# Model Comparison Report

**Project:** Cloud Cost Anomaly Detection Platform
**Phase:** 1.7 — Model Comparison
**Evaluation scope:** `full_dataset_outputs`

---

## Overview

This report consolidates a side-by-side comparison of all four anomaly detectors
on the full 10,800-record synthetic billing dataset.

**Scope note:** All metrics below are computed on the **complete prediction output**
of each model (`evaluation_scope = full_dataset_outputs`).
Decision Tree and XGBoost had their primary metrics computed on held-out test splits
(80/20) in Phases 1.5 and 1.6. Those per-phase test-split results are the correct
reference for unbiased supervised model evaluation. This report provides an
operational view across the full dataset for comparison purposes.

---

## Summary

| model_name | model_type | evaluation_scope | anomaly_count | anomaly_rate | precision | recall | f1_score | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Statistical Baseline | rule_based | full_dataset_outputs | 573 | 0.0531 | 0.8988 | 0.9537 | 0.9254 | 0.9749 |
| Isolation Forest | unsupervised_ml | full_dataset_outputs | 540 | 0.05 | 0.4944 | 0.4944 | 0.4944 | 0.9394 |
| Decision Tree | supervised_ml | full_dataset_outputs | 623 | 0.0577 | 0.8331 | 0.9611 | 0.8925 | 0.9811 |
| XGBoost | supervised_ml | full_dataset_outputs | 566 | 0.0524 | 0.9452 | 0.9907 | 0.9675 | 0.9997 |

**Best F1:** XGBoost (0.9675)
**Best Recall:** XGBoost (0.9907)

---

## Detailed Counts

| model_name | true_positives | false_positives | false_negatives | true_negatives | accuracy |
| --- | --- | --- | --- | --- | --- |
| Statistical Baseline | 515 | 58 | 25 | 10202 | 0.9923 |
| Isolation Forest | 267 | 273 | 273 | 9987 | 0.9494 |
| Decision Tree | 519 | 104 | 21 | 10156 | 0.9884 |
| XGBoost | 535 | 31 | 5 | 10229 | 0.9967 |

---

## Model Interpretation

### Statistical Baseline
Rule-based detector applying fixed thresholds on rolling cost features and FinOps
governance flags (missing tags, cost spikes above 7d/30d averages). No ML involved.
Strong performance on synthetic data because its thresholds were designed for this
specific schema. In production, thresholds require continuous tuning as cost patterns
evolve.

### Isolation Forest
Unsupervised ML — labels were never used during training. Isolates anomalies by
randomly partitioning the feature space. Evaluated here on the full dataset
including its own training data (no holdout). Its performance reflects the inherent
difficulty of unsupervised detection without label guidance.

### Decision Tree
Supervised ML classifier with constrained depth for explainability. Trained on 80%
of the dataset; original test-split metrics from Phase 1.5. Full-dataset evaluation
in this report includes training data, which inflates apparent performance relative
to the Phase 1.5 test-split results.

### XGBoost
Principal supervised ML classifier using gradient-boosted trees. Trained on 80% of
the dataset; original test-split metrics from Phase 1.6. Expected to achieve the
best overall performance. Full-dataset evaluation here includes training data.

---

## Important Notes

1. **Synthetic dataset.** All metrics are computed on synthetic billing data
   generated with programmatic anomaly injection. Real billing data has no
   ground-truth labels and different statistical properties.

2. **evaluation_scope = full_dataset_outputs.** This report evaluates the complete
   output of each model. Supervised models (Decision Tree, XGBoost) were also
   evaluated on held-out test splits in their respective phases — those results
   are the correct reference for unbiased model selection.

3. **Controlled benchmark only.** These metrics are not a guarantee of production
   performance. Real deployment requires real billing data, continuous validation,
   and monitoring for concept drift.

4. **Baseline advantage in synthetic setting.** The Statistical Baseline was
   calibrated for this synthetic schema. In production, its fixed thresholds would
   need tuning as cost distributions change over time.
