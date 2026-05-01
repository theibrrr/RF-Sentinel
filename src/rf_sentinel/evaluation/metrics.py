"""Classification and confidence metrics computation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from rf_sentinel.utils.logging import print_info
from rf_sentinel.utils.paths import ensure_dir


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute simple classification accuracy without sklearn target warnings."""
    return float(np.mean(y_true == y_pred))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute comprehensive classification metrics."""
    labels = list(range(len(class_names)))
    acc = _accuracy(y_true, y_pred)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "per_class_report": report,
    }


def compute_confidence_metrics(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.70,
) -> dict:
    """Compute confidence-aware metrics."""
    confidences = np.max(probabilities, axis=1)
    accepted_mask = confidences >= threshold
    uncertain_mask = ~accepted_mask

    n_total = len(y_true)
    n_accepted = int(np.sum(accepted_mask))
    n_uncertain = int(np.sum(uncertain_mask))

    metrics = {
        "confidence_threshold": threshold,
        "total_samples": n_total,
        "accepted_count": n_accepted,
        "uncertain_count": n_uncertain,
        "accepted_ratio": n_accepted / max(n_total, 1),
        "uncertain_ratio": n_uncertain / max(n_total, 1),
        "mean_confidence": float(np.mean(confidences)),
        "median_confidence": float(np.median(confidences)),
        "std_confidence": float(np.std(confidences)),
    }

    if n_accepted > 0:
        metrics["accepted_accuracy"] = _accuracy(y_true[accepted_mask], y_pred[accepted_mask])
    else:
        metrics["accepted_accuracy"] = 0.0

    if n_uncertain > 0:
        metrics["uncertain_accuracy"] = _accuracy(y_true[uncertain_mask], y_pred[uncertain_mask])
    else:
        metrics["uncertain_accuracy"] = 0.0

    return metrics


def save_evaluation_metrics(metrics: dict, report_dir: str | Path) -> None:
    """Save evaluation metrics as JSON."""
    report_dir = ensure_dir(report_dir)
    path = report_dir / "evaluation_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    print_info(f"Saved evaluation metrics: {path}")


def save_classification_report_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    report_dir: str | Path,
) -> None:
    """Save sklearn classification report as CSV."""
    report_dir = ensure_dir(report_dir)
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).transpose()
    path = report_dir / "classification_report.csv"
    df.to_csv(path)
    print_info(f"Saved classification report: {path}")
