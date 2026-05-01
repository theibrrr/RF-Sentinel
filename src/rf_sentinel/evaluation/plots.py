"""Evaluation plot generation — confusion matrix, SNR curves, histograms."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rf_sentinel.utils.logging import print_info


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    normalize: bool = True,
) -> None:
    """Plot and save confusion matrix."""
    plt = _setup_matplotlib()

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums > 0,
        )
    else:
        cm_norm = cm

    n = len(class_names)
    figsize = max(8, n * 0.5)
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print_info(f"Saved confusion matrix: {save_path}")


def plot_accuracy_vs_snr(
    snr_values: list[float],
    accuracies: list[float],
    save_path: str | Path,
) -> None:
    """Plot accuracy vs SNR curve."""
    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(snr_values, accuracies, "o-", color="#2196F3", linewidth=2, markersize=6)
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Classification Accuracy vs SNR", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print_info(f"Saved accuracy vs SNR plot: {save_path}")


def plot_confidence_histogram(
    confidences: np.ndarray,
    threshold: float,
    save_path: str | Path,
) -> None:
    """Plot confidence score histogram with threshold line."""
    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confidences, bins=50, color="#4CAF50", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(
        x=threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.2f}"
    )
    ax.set_xlabel("Confidence (max probability)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Prediction Confidence Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print_info(f"Saved confidence histogram: {save_path}")
