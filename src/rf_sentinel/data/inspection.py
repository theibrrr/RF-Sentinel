"""Dataset inspection utilities — generate summary reports."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from rf_sentinel.utils.logging import print_info, print_warning
from rf_sentinel.utils.paths import ensure_dir


def inspect_hdf5(dataset_path: str | Path, cfg: dict) -> dict:
    """Inspect an HDF5 dataset and return a summary dictionary."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data_cfg = cfg.get("data", {})
    x_key = data_cfg.get("x_key", "X")
    y_key = data_cfg.get("y_key", "Y")
    snr_key = data_cfg.get("snr_key", "Z")

    summary = {"file_path": str(dataset_path), "file_size_mb": dataset_path.stat().st_size / 1e6}

    with h5py.File(str(dataset_path), "r") as f:
        keys_info = {}
        for key in f:
            ds = f[key]
            keys_info[key] = {"shape": list(ds.shape), "dtype": str(ds.dtype)}
        summary["hdf5_keys"] = keys_info

        if x_key in f:
            x_dataset = f[x_key]
            summary["num_samples"] = x_dataset.shape[0]
            summary["sample_shape"] = list(x_dataset.shape[1:])
            summary["x_dtype"] = str(x_dataset.dtype)

        if y_key in f:
            y_raw = f[y_key][:]
            if y_raw.ndim == 2 and y_raw.shape[1] > 1:
                labels = np.argmax(y_raw, axis=1)
                summary["label_encoding"] = "one-hot"
                summary["num_classes"] = y_raw.shape[1]
            else:
                labels = y_raw.reshape(-1).astype(int)
                summary["label_encoding"] = "integer"
                summary["num_classes"] = int(labels.max()) + 1

            class_counts = Counter(labels.tolist())
            summary["samples_per_class"] = dict(sorted(class_counts.items()))

        if snr_key in f:
            snr = f[snr_key][:].flatten()
            unique_snr = sorted(set(snr.tolist()))
            summary["snr_range"] = [float(min(unique_snr)), float(max(unique_snr))]
            summary["unique_snr_values"] = [float(s) for s in unique_snr]
            summary["num_unique_snr"] = len(unique_snr)

            snr_counts = Counter(snr.astype(int).tolist())
            summary["samples_per_snr"] = dict(sorted(snr_counts.items()))

            if "samples_per_class" in summary:
                class_snr_counts: dict[str, dict[str, int]] = {}
                for label, snr_value in zip(labels, snr.astype(int), strict=True):
                    class_key = str(int(label))
                    snr_key_str = str(int(snr_value))
                    class_snr_counts.setdefault(class_key, {})
                    class_snr_counts[class_key][snr_key_str] = (
                        class_snr_counts[class_key].get(snr_key_str, 0) + 1
                    )
                summary["samples_per_class_snr"] = {
                    class_key: dict(
                        sorted(snr_counts_for_class.items(), key=lambda item: int(item[0]))
                    )
                    for class_key, snr_counts_for_class in sorted(
                        class_snr_counts.items(), key=lambda item: int(item[0])
                    )
                }

    # Warnings
    warnings = []
    if "samples_per_class" in summary:
        counts = list(summary["samples_per_class"].values())
        if max(counts) > 2 * min(counts):
            warnings.append("Significant class imbalance detected.")
    summary["warnings"] = warnings

    return summary


def save_summary_json(summary: dict, report_dir: str | Path) -> Path:
    """Save summary as JSON."""
    report_dir = ensure_dir(report_dir)
    path = report_dir / "dataset_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print_info(f"Saved dataset summary JSON: {path}")
    return path


def save_summary_markdown(summary: dict, report_dir: str | Path) -> Path:
    """Save summary as Markdown."""
    report_dir = ensure_dir(report_dir)
    path = report_dir / "dataset_summary.md"

    lines = ["# Dataset Summary\n"]
    lines.append(f"**File**: `{summary.get('file_path', 'N/A')}`\n")
    lines.append(f"**Size**: {summary.get('file_size_mb', 0):.1f} MB\n")
    lines.append(f"**Samples**: {summary.get('num_samples', 'N/A')}\n")
    lines.append(f"**Sample shape**: {summary.get('sample_shape', 'N/A')}\n")
    lines.append(f"**Classes**: {summary.get('num_classes', 'N/A')}\n")
    lines.append(f"**Label encoding**: {summary.get('label_encoding', 'N/A')}\n")

    if "snr_range" in summary:
        lines.append(f"**SNR range**: {summary['snr_range'][0]} to {summary['snr_range'][1]} dB\n")
        lines.append(f"**Unique SNR values**: {summary.get('num_unique_snr', 'N/A')}\n")

    lines.append("\n## HDF5 Keys\n")
    for key, info in summary.get("hdf5_keys", {}).items():
        lines.append(f"- `{key}`: shape={info['shape']}, dtype={info['dtype']}\n")

    if "samples_per_class" in summary:
        lines.append("\n## Samples per Class\n")
        lines.append("| Class | Count |\n|-------|-------|\n")
        for cls, count in summary["samples_per_class"].items():
            lines.append(f"| {cls} | {count} |\n")

    if "samples_per_snr" in summary:
        lines.append("\n## Samples per SNR\n")
        lines.append("| SNR (dB) | Count |\n|----------|-------|\n")
        for snr, count in summary["samples_per_snr"].items():
            lines.append(f"| {snr} | {count} |\n")

    if summary.get("warnings"):
        lines.append("\n## Warnings\n")
        for w in summary["warnings"]:
            lines.append(f"- WARNING: {w}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print_info(f"Saved dataset summary Markdown: {path}")
    return path


def run_inspection(cfg: dict) -> dict:
    """Run full dataset inspection and save reports."""
    data_cfg = cfg["data"]
    dataset_path = data_cfg["dataset_path"]
    report_dir = cfg.get("project", {}).get("report_dir", "reports")

    print_info("Running dataset inspection...")
    summary = inspect_hdf5(dataset_path, cfg)

    save_summary_json(summary, report_dir)
    save_summary_markdown(summary, report_dir)

    # Print summary to console
    print_info(f"Samples: {summary.get('num_samples', 'N/A')}")
    print_info(f"Classes: {summary.get('num_classes', 'N/A')}")
    if "snr_range" in summary:
        print_info(f"SNR range: {summary['snr_range'][0]} to {summary['snr_range'][1]} dB")
    for w in summary.get("warnings", []):
        print_warning(w)

    return summary
