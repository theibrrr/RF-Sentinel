"""Low-SNR failure and error analysis with markdown report generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rf_sentinel.utils.logging import print_info
from rf_sentinel.utils.paths import ensure_dir


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy without sklearn target-type warnings on tiny subsets."""
    return float(np.mean(y_true == y_pred))


def find_most_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    top_n: int = 10,
) -> list[dict]:
    """Find the most confused class pairs from the confusion matrix."""
    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for true_label, predicted_label in zip(y_true, y_pred, strict=True):
        true_idx = int(true_label)
        pred_idx = int(predicted_label)
        if 0 <= true_idx < len(class_names) and 0 <= pred_idx < len(class_names):
            cm[true_idx, pred_idx] += 1

    np.fill_diagonal(cm, 0)

    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] > 0:
                pairs.append(
                    {
                        "true_class": class_names[i],
                        "predicted_class": class_names[j],
                        "count": int(cm[i, j]),
                    }
                )

    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_n]


def compute_per_class_snr_degradation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr: np.ndarray,
    class_names: list[str],
    low_snr_max: float = 0,
    high_snr_min: float = 12,
) -> list[dict]:
    """Compute per-class accuracy degradation from high-SNR to low-SNR."""
    low_mask = snr <= low_snr_max
    high_mask = snr >= high_snr_min

    degradation = []
    for i, name in enumerate(class_names):
        class_mask = y_true == i

        low_class = class_mask & low_mask
        high_class = class_mask & high_mask

        low_acc = _accuracy(y_true[low_class], y_pred[low_class]) if low_class.sum() > 0 else None

        high_acc = (
            _accuracy(y_true[high_class], y_pred[high_class]) if high_class.sum() > 0 else None
        )

        delta = None
        if low_acc is not None and high_acc is not None:
            delta = high_acc - low_acc

        degradation.append(
            {
                "class": name,
                "low_snr_accuracy": low_acc,
                "high_snr_accuracy": high_acc,
                "degradation": delta,
            }
        )

    degradation.sort(key=lambda x: x.get("degradation") or 0, reverse=True)
    return degradation


def generate_error_analysis_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr: np.ndarray,
    class_names: list[str],
    cfg: dict,
    report_dir: str | Path,
) -> Path:
    """Generate a comprehensive markdown error analysis report."""
    report_dir = ensure_dir(report_dir)
    eval_cfg = cfg.get("evaluation", {})
    low_snr_max = eval_cfg.get("low_snr_max", 0)
    high_snr_min = eval_cfg.get("high_snr_min", 12)

    lines = ["# Error Analysis Report\n\n"]

    # Low-SNR confused pairs
    low_mask = snr <= low_snr_max
    if low_mask.sum() > 0:
        lines.append(f"## Most Confused Class Pairs (SNR <= {low_snr_max} dB)\n\n")
        pairs = find_most_confused_pairs(
            y_true[low_mask],
            y_pred[low_mask],
            class_names,
            top_n=15,
        )
        lines.append("| True Class | Predicted Class | Misclassification Count |\n")
        lines.append("|------------|-----------------|------------------------|\n")
        for p in pairs:
            lines.append(f"| {p['true_class']} | {p['predicted_class']} | {p['count']} |\n")
        lines.append("\n")

    # Per-class degradation
    lines.append("## Per-Class SNR Degradation\n\n")
    lines.append(f"Comparing high-SNR (>={high_snr_min} dB) vs low-SNR (<={low_snr_max} dB).\n\n")
    degradation = compute_per_class_snr_degradation(
        y_true,
        y_pred,
        snr,
        class_names,
        low_snr_max,
        high_snr_min,
    )
    lines.append("| Class | High-SNR Acc | Low-SNR Acc | Degradation |\n")
    lines.append("|-------|-------------|-------------|-------------|\n")
    for d in degradation:
        high = f"{d['high_snr_accuracy']:.3f}" if d["high_snr_accuracy"] is not None else "N/A"
        low = f"{d['low_snr_accuracy']:.3f}" if d["low_snr_accuracy"] is not None else "N/A"
        deg = f"{d['degradation']:.3f}" if d["degradation"] is not None else "N/A"
        lines.append(f"| {d['class']} | {high} | {low} | {deg} |\n")
    lines.append("\n")

    # Overall confusion
    lines.append("## Overall Most Confused Pairs\n\n")
    pairs_all = find_most_confused_pairs(y_true, y_pred, class_names, top_n=10)
    lines.append("| True Class | Predicted Class | Count |\n")
    lines.append("|------------|-----------------|-------|\n")
    for p in pairs_all:
        lines.append(f"| {p['true_class']} | {p['predicted_class']} | {p['count']} |\n")

    path = report_dir / "error_analysis.md"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print_info(f"Saved error analysis report: {path}")
    return path
