from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.compare_models import (
    compare_all_models,
    compute_model_metrics,
    save_csv,
    save_markdown,
)

# ---------------------------------------------------------------------------
# Fixture: 20 records, 2 anomalies — all prediction columns present
# ---------------------------------------------------------------------------
#
# Baseline: TP=2, FP=1  (extra detection at pos 17)
# IForest : TP=2, FP=0  (perfect)
# DT      : TP=1, FP=0, FN=1  (misses pos 18)
# XGBoost : TP=2, FP=0  (perfect)

@pytest.fixture
def mock_df() -> pd.DataFrame:
    return pd.DataFrame({
        "is_anomaly": [False] * 18 + [True, True],
        "baseline_anomaly": [False] * 17 + [True, True, True],
        "baseline_score":   [0.1] * 17  + [0.8, 0.8, 0.8],
        "iforest_anomaly":  [False] * 18 + [True, True],
        "iforest_score":    [0.1] * 18  + [0.9, 0.9],
        "dt_anomaly":       [False] * 19 + [True],
        "dt_score":         [0.1] * 19  + [0.8],
        "xgb_anomaly":      [False] * 18 + [True, True],
        "xgb_score":        [0.1] * 18  + [0.95, 0.85],
    })


# ---------------------------------------------------------------------------
# compute_model_metrics
# ---------------------------------------------------------------------------

class TestComputeModelMetrics:
    def test_returns_required_keys(self, mock_df):
        r = compute_model_metrics(mock_df, "baseline_anomaly", "baseline_score",
                                  "Test Model", "rule_based", "notes")
        for k in ("model_name", "model_type", "evaluation_scope", "anomaly_count",
                  "anomaly_rate", "true_positives", "false_positives",
                  "false_negatives", "true_negatives", "accuracy",
                  "precision", "recall", "f1_score", "roc_auc", "notes"):
            assert k in r, f"Missing key: {k}"

    def test_evaluation_scope_is_full_dataset_outputs(self, mock_df):
        r = compute_model_metrics(mock_df, "baseline_anomaly", "baseline_score",
                                  "B", "rule_based", "")
        assert r["evaluation_scope"] == "full_dataset_outputs"

    def test_anomaly_count_correct(self, mock_df):
        r = compute_model_metrics(mock_df, "baseline_anomaly", "baseline_score",
                                  "B", "rule_based", "")
        assert r["anomaly_count"] == 3  # 2 TP + 1 FP

    def test_tp_fp_fn_tn_for_baseline(self, mock_df):
        r = compute_model_metrics(mock_df, "baseline_anomaly", "baseline_score",
                                  "B", "rule_based", "")
        assert r["true_positives"] == 2
        assert r["false_positives"] == 1
        assert r["false_negatives"] == 0
        assert r["true_negatives"] == 17

    def test_tp_fp_fn_tn_for_iforest(self, mock_df):
        r = compute_model_metrics(mock_df, "iforest_anomaly", "iforest_score",
                                  "IF", "unsupervised_ml", "")
        assert r["true_positives"] == 2
        assert r["false_positives"] == 0
        assert r["false_negatives"] == 0
        assert r["true_negatives"] == 18

    def test_perfect_precision_and_recall(self, mock_df):
        r = compute_model_metrics(mock_df, "iforest_anomaly", "iforest_score",
                                  "IF", "unsupervised_ml", "")
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(1.0)
        assert r["f1_score"] == pytest.approx(1.0)

    def test_partial_recall_for_dt(self, mock_df):
        r = compute_model_metrics(mock_df, "dt_anomaly", "dt_score",
                                  "DT", "supervised_ml", "")
        assert r["recall"] == pytest.approx(0.5)
        assert r["precision"] == pytest.approx(1.0)

    def test_anomaly_rate_correct(self, mock_df):
        r = compute_model_metrics(mock_df, "baseline_anomaly", "baseline_score",
                                  "B", "rule_based", "")
        assert r["anomaly_rate"] == pytest.approx(3 / 20)

    def test_model_name_and_notes_preserved(self, mock_df):
        r = compute_model_metrics(mock_df, "xgb_anomaly", "xgb_score",
                                  "XGBoost", "supervised_ml", "my note")
        assert r["model_name"] == "XGBoost"
        assert r["model_type"] == "supervised_ml"
        assert r["notes"] == "my note"

    def test_roc_auc_in_range(self, mock_df):
        r = compute_model_metrics(mock_df, "xgb_anomaly", "xgb_score",
                                  "XGB", "supervised_ml", "")
        assert 0.0 <= r["roc_auc"] <= 1.0


# ---------------------------------------------------------------------------
# compare_all_models
# ---------------------------------------------------------------------------

class TestCompareAllModels:
    def test_returns_dataframe(self, mock_df):
        result = compare_all_models(mock_df, mock_df, mock_df, mock_df)
        assert isinstance(result, pd.DataFrame)

    def test_exactly_four_rows(self, mock_df):
        result = compare_all_models(mock_df, mock_df, mock_df, mock_df)
        assert len(result) == 4

    def test_required_columns_present(self, mock_df):
        result = compare_all_models(mock_df, mock_df, mock_df, mock_df)
        for col in ("model_name", "model_type", "evaluation_scope",
                    "anomaly_count", "precision", "recall", "f1_score", "notes"):
            assert col in result.columns

    def test_all_model_names_present(self, mock_df):
        result = compare_all_models(mock_df, mock_df, mock_df, mock_df)
        names = set(result["model_name"])
        assert "Statistical Baseline" in names
        assert "Isolation Forest" in names
        assert "Decision Tree" in names
        assert "XGBoost" in names

    def test_all_rows_have_full_dataset_scope(self, mock_df):
        result = compare_all_models(mock_df, mock_df, mock_df, mock_df)
        assert (result["evaluation_scope"] == "full_dataset_outputs").all()

    def test_model_types_correct(self, mock_df):
        result = compare_all_models(mock_df, mock_df, mock_df, mock_df)
        types = dict(zip(result["model_name"], result["model_type"]))
        assert types["Statistical Baseline"] == "rule_based"
        assert types["Isolation Forest"] == "unsupervised_ml"
        assert types["Decision Tree"] == "supervised_ml"
        assert types["XGBoost"] == "supervised_ml"


# ---------------------------------------------------------------------------
# save_csv
# ---------------------------------------------------------------------------

class TestSaveCSV:
    def test_csv_file_created(self, tmp_path, mock_df):
        comparison = compare_all_models(mock_df, mock_df, mock_df, mock_df)
        path = str(tmp_path / "test.csv")
        save_csv(comparison, path)
        assert Path(path).exists()

    def test_csv_has_four_rows(self, tmp_path, mock_df):
        comparison = compare_all_models(mock_df, mock_df, mock_df, mock_df)
        path = str(tmp_path / "test.csv")
        save_csv(comparison, path)
        df = pd.read_csv(path)
        assert len(df) == 4

    def test_csv_has_evaluation_scope_column(self, tmp_path, mock_df):
        comparison = compare_all_models(mock_df, mock_df, mock_df, mock_df)
        path = str(tmp_path / "test.csv")
        save_csv(comparison, path)
        df = pd.read_csv(path)
        assert "evaluation_scope" in df.columns
        assert (df["evaluation_scope"] == "full_dataset_outputs").all()

    def test_csv_creates_parent_dir(self, tmp_path, mock_df):
        comparison = compare_all_models(mock_df, mock_df, mock_df, mock_df)
        path = str(tmp_path / "nested" / "dir" / "test.csv")
        save_csv(comparison, path)
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# save_markdown
# ---------------------------------------------------------------------------

class TestSaveMarkdown:
    def test_md_file_created(self, tmp_path, mock_df):
        comparison = compare_all_models(mock_df, mock_df, mock_df, mock_df)
        path = str(tmp_path / "test.md")
        save_markdown(comparison, path)
        assert Path(path).exists()

    def test_md_contains_all_model_names(self, tmp_path, mock_df):
        comparison = compare_all_models(mock_df, mock_df, mock_df, mock_df)
        path = str(tmp_path / "test.md")
        save_markdown(comparison, path)
        content = Path(path).read_text()
        assert "Statistical Baseline" in content
        assert "Isolation Forest" in content
        assert "Decision Tree" in content
        assert "XGBoost" in content

    def test_md_contains_evaluation_scope(self, tmp_path, mock_df):
        comparison = compare_all_models(mock_df, mock_df, mock_df, mock_df)
        path = str(tmp_path / "test.md")
        save_markdown(comparison, path)
        content = Path(path).read_text()
        assert "full_dataset_outputs" in content

    def test_md_contains_honesty_note_about_synthetic(self, tmp_path, mock_df):
        comparison = compare_all_models(mock_df, mock_df, mock_df, mock_df)
        path = str(tmp_path / "test.md")
        save_markdown(comparison, path)
        content = Path(path).read_text()
        assert "synthetic" in content.lower()

    def test_md_contains_table_separator(self, tmp_path, mock_df):
        comparison = compare_all_models(mock_df, mock_df, mock_df, mock_df)
        path = str(tmp_path / "test.md")
        save_markdown(comparison, path)
        content = Path(path).read_text()
        assert "---" in content

    def test_md_creates_parent_dir(self, tmp_path, mock_df):
        comparison = compare_all_models(mock_df, mock_df, mock_df, mock_df)
        path = str(tmp_path / "reports" / "test.md")
        save_markdown(comparison, path)
        assert Path(path).exists()
