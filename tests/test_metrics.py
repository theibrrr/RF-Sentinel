"""Tests for evaluation metrics."""

import numpy as np

from rf_sentinel.evaluation.metrics import compute_classification_metrics


def test_classification_metrics_include_missing_classes():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1, 0])
    class_names = ["A", "B", "C", "D"]

    metrics = compute_classification_metrics(y_true, y_pred, class_names)

    assert metrics["accuracy"] == 2 / 3
    assert np.array(metrics["confusion_matrix"]).shape == (4, 4)
    assert "C" in metrics["per_class_report"]
    assert "D" in metrics["per_class_report"]
