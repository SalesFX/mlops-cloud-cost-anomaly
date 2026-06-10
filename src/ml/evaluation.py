"""Shared evaluation utilities for supervised binary classifiers.

Used by decision_tree_classifier and xgboost_classifier to compute
classification metrics on a held-out test set.

Evaluation on synthetic data is a controlled benchmark, not a proxy for
production performance on real unlabeled billing data (Phase 6).
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    """Compute classification metrics from arrays.

    Decoupled from any DataFrame — takes raw arrays so it can be reused
    by any supervised classifier without assumptions about column names.

    Args:
        y_true:  Ground-truth labels (int or bool, 0=normal 1=anomaly).
        y_pred:  Predicted binary labels (int or bool).
        y_score: Predicted probability for the anomaly class (float [0, 1]).

    Returns:
        dict with keys: tp, fp, fn, tn, accuracy, precision, recall, f1, roc_auc.
    """
    actual = y_true.astype(bool)
    predicted = y_pred.astype(bool)

    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())
    n = tp + fp + fn + tn

    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    try:
        roc_auc = float(roc_auc_score(actual.astype(int), y_score))
    except ValueError:
        roc_auc = 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
    }
