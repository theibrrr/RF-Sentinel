"""SNR-stratified robustness analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rf_sentinel.utils.logging import print_info
from rf_sentinel.utils.paths import ensure_dir


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute simple classification accuracy without sklearn target warnings."""
    return float(np.mean(y_true == y_pred))


def compute_per_snr_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr: np.ndarray,
) -> dict[float, float]:
    """Compute accuracy for each unique SNR value."""
    unique_snr = sorted(set(snr.tolist()))
    per_snr = {}
    for s in unique_snr:
        mask = snr == s
        if mask.sum() > 0:
            per_snr[float(s)] = _accuracy(y_true[mask], y_pred[mask])
    return per_snr


def compute_snr_band_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr: np.ndarray,
    low_snr_max: float = 0,
    high_snr_min: float = 12,
) -> dict:
    """Compute metrics for low-SNR and high-SNR subsets."""
    low_mask = snr <= low_snr_max
    high_mask = snr >= high_snr_min

    result = {
        "low_snr_max": low_snr_max,
        "high_snr_min": high_snr_min,
    }

    if low_mask.sum() > 0:
        result["low_snr_accuracy"] = _accuracy(y_true[low_mask], y_pred[low_mask])
        result["low_snr_samples"] = int(low_mask.sum())
    else:
        result["low_snr_accuracy"] = None
        result["low_snr_samples"] = 0

    if high_mask.sum() > 0:
        result["high_snr_accuracy"] = _accuracy(y_true[high_mask], y_pred[high_mask])
        result["high_snr_samples"] = int(high_mask.sum())
    else:
        result["high_snr_accuracy"] = None
        result["high_snr_samples"] = 0

    return result


def save_per_snr_accuracy_csv(
    per_snr: dict[float, float],
    report_dir: str | Path,
) -> None:
    """Save per-SNR accuracy as CSV."""
    report_dir = ensure_dir(report_dir)
    df = pd.DataFrame(list(per_snr.items()), columns=["SNR_dB", "Accuracy"])
    df = df.sort_values("SNR_dB").reset_index(drop=True)
    path = report_dir / "per_snr_accuracy.csv"
    df.to_csv(path, index=False)
    print_info(f"Saved per-SNR accuracy: {path}")
