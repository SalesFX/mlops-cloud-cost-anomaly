import numpy as np
import pytest

from src.ml.evaluation import evaluate_binary_classifier


class TestEvaluateBinaryClassifier:
    def test_returns_required_keys(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.6, 0.3, 0.1])
        r = evaluate_binary_classifier(y_true, y_pred, y_score)
        for k in ("tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1", "roc_auc"):
            assert k in r

    def test_correct_counts(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.6, 0.3, 0.1])
        r = evaluate_binary_classifier(y_true, y_pred, y_score)
        assert r["tp"] == 1
        assert r["fp"] == 1
        assert r["fn"] == 1
        assert r["tn"] == 1

    def test_precision_recall_f1(self):
        r = evaluate_binary_classifier(
            np.array([1, 0, 1, 0]),
            np.array([1, 1, 0, 0]),
            np.array([0.9, 0.6, 0.3, 0.1]),
        )
        assert r["precision"] == pytest.approx(0.5)
        assert r["recall"] == pytest.approx(0.5)
        assert r["f1"] == pytest.approx(0.5)

    def test_perfect_predictions(self):
        r = evaluate_binary_classifier(
            np.array([1, 0, 1, 0]),
            np.array([1, 0, 1, 0]),
            np.array([0.9, 0.1, 0.8, 0.2]),
        )
        assert r["accuracy"] == pytest.approx(1.0)
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(1.0)
        assert r["f1"] == pytest.approx(1.0)

    def test_roc_auc_in_range(self):
        r = evaluate_binary_classifier(
            np.array([1, 0, 1, 0]),
            np.array([1, 1, 0, 0]),
            np.array([0.9, 0.6, 0.3, 0.1]),
        )
        assert 0.0 <= r["roc_auc"] <= 1.0

    def test_zero_division_safe(self):
        r = evaluate_binary_classifier(
            np.array([1, 0]),
            np.array([0, 0]),
            np.array([0.3, 0.2]),
        )
        assert r["precision"] == pytest.approx(0.0)
        assert r["f1"] == pytest.approx(0.0)

    def test_roc_auc_graceful_single_class(self):
        r = evaluate_binary_classifier(
            np.array([0, 0, 0]),
            np.array([1, 0, 1]),
            np.array([0.8, 0.3, 0.7]),
        )
        assert r["roc_auc"] == pytest.approx(0.0)

    def test_accepts_bool_arrays(self):
        r = evaluate_binary_classifier(
            np.array([True, False, True, False]),
            np.array([True, False, True, False]),
            np.array([0.9, 0.1, 0.8, 0.2]),
        )
        assert r["accuracy"] == pytest.approx(1.0)
