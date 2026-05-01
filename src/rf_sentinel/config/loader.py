"""YAML configuration loader with validation and path expansion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Default RadioML 2018.01A class names (24 modulation types)
RADIOML_CLASS_NAMES: list[str] = [
    "OOK",
    "4ASK",
    "8ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "16APSK",
    "32APSK",
    "64APSK",
    "128APSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "AM-SSB-WC",
    "AM-SSB-SC",
    "AM-DSB-WC",
    "AM-DSB-SC",
    "FM",
    "GMSK",
    "OQPSK",
]


_DEFAULTS: dict[str, Any] = {
    "project": {
        "name": "RF-Sentinel",
        "seed": 42,
        "output_dir": "artifacts",
        "report_dir": "reports",
    },
    "data": {
        "dataset_path": "data/raw/GOLD_XYZ_OSC.0001_1024.hdf5",
        "x_key": "X",
        "y_key": "Y",
        "snr_key": "Z",
        "sample_length": 1024,
        "iq_channels": 2,
        "max_samples": None,
        "subset_fraction": None,
        "normalize": "rms",
        "split_dir": "data/splits",
        "reuse_splits": True,
    },
    "splits": {
        "train_size": 0.70,
        "val_size": 0.15,
        "test_size": 0.15,
        "stratify_by": "label_snr",
        "random_seed": 42,
    },
    "model": {
        "type": "cnn1d",
        "input_channels": 2,
        "num_classes": "auto",
        "dropout": 0.25,
        "base_channels": 64,
    },
    "training": {
        "epochs": 30,
        "batch_size": 256,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "optimizer": "adam",
        "scheduler": "reduce_on_plateau",
        "early_stopping_patience": 7,
        "num_workers": 2,
        "device": "auto",
        "mixed_precision": False,
    },
    "evaluation": {
        "confidence_threshold": 0.70,
        "top_k": 3,
        "low_snr_max": 0,
        "high_snr_min": 12,
        "save_confusion_matrix": True,
        "save_snr_curve": True,
    },
    "mlflow": {
        "enabled": True,
        "tracking_uri": "mlruns",
        "experiment_name": "rf-sentinel-waveform-models",
        "run_name": "cnn1d-baseline",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand_paths(cfg: dict, base_dir: Path) -> dict:
    """Expand relative paths in known path-like fields."""
    path_fields = [
        ("data", "dataset_path"),
        ("data", "split_dir"),
        ("project", "output_dir"),
        ("project", "report_dir"),
    ]
    for section, key in path_fields:
        if section in cfg and key in cfg[section]:
            val = cfg[section][key]
            if val and not os.path.isabs(val):
                cfg[section][key] = str(base_dir / val)
    return cfg


def _apply_env_overrides(cfg: dict) -> dict:
    """Apply environment variable overrides."""
    env_dataset = os.environ.get("RF_SENTINEL_DATASET_PATH")
    if env_dataset:
        cfg.setdefault("data", {})["dataset_path"] = env_dataset
    return cfg


def validate_config(cfg: dict) -> list[str]:
    """Validate configuration and return list of warnings."""
    warnings = []
    required_sections = ["project", "data", "model"]
    for section in required_sections:
        if section not in cfg:
            warnings.append(f"Missing required config section: '{section}'")

    if "splits" in cfg:
        splits = cfg["splits"]
        total = splits.get("train_size", 0) + splits.get("val_size", 0) + splits.get("test_size", 0)
        if abs(total - 1.0) > 0.01:
            warnings.append(
                f"Split sizes sum to {total:.2f}, expected ~1.0. "
                f"Got train={splits.get('train_size')}, val={splits.get('val_size')}, "
                f"test={splits.get('test_size')}"
            )

    if "training" in cfg:
        lr = cfg["training"].get("learning_rate", 0)
        if lr <= 0:
            warnings.append(f"Learning rate must be positive, got {lr}")

    if "model" in cfg:
        model_cfg = cfg["model"]
        model_type = str(model_cfg.get("type", "cnn1d")).lower()
        allowed = {"cnn1d", "resnet1d", "tcn1d", "xgboost"}
        if model_type not in allowed:
            warnings.append(
                f"Unknown model type '{model_type}'. Supported types: {sorted(allowed)}"
            )
        if model_type == "tcn1d" and model_cfg.get("kernel_size", 5) % 2 == 0:
            warnings.append("TCN1D kernel_size should be odd to preserve sequence length.")

    return warnings


def load_config(
    config_path: str | Path,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load YAML configuration with defaults, validation, and path expansion.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.
    cli_overrides : dict, optional
        Additional overrides applied on top of the loaded config.

    Returns
    -------
    dict
        Fully resolved configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If validation finds critical errors.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    # Merge: defaults <- user yaml <- cli overrides
    cfg = _deep_merge(_DEFAULTS, user_cfg)
    if cli_overrides:
        cfg = _deep_merge(cfg, cli_overrides)

    # Apply environment variable overrides
    cfg = _apply_env_overrides(cfg)

    # Expand relative paths
    base_dir = Path.cwd()
    cfg = _expand_paths(cfg, base_dir)

    # Validate
    warnings = validate_config(cfg)
    for w in warnings:
        print(f"[CONFIG WARNING] {w}")

    return cfg
