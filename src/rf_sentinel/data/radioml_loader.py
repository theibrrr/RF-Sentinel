"""RadioML 2018.01A HDF5 dataset loader.

Loads raw I/Q samples, one-hot labels, and SNR values from the DeepSig
RadioML 2018.01A HDF5 file. Provides validation, label decoding, and
optional subsetting for development workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from rf_sentinel.config.loader import RADIOML_CLASS_NAMES
from rf_sentinel.utils.logging import get_logger, print_info, print_warning

logger = get_logger(__name__)


class RadioMLDataset:
    """Container for loaded RadioML 2018.01A data.

    Attributes
    ----------
    X : np.ndarray
        Raw I/Q samples, shape (N, 1024, 2).
    labels : np.ndarray
        Integer class labels, shape (N,).
    snr : np.ndarray
        SNR values in dB, shape (N,).
    class_names : list[str]
        Ordered modulation class names.
    num_classes : int
        Number of unique modulation classes.
    label_to_index : dict[str, int]
        Mapping from class name to index.
    index_to_label : dict[int, str]
        Mapping from index to class name.
    """

    def __init__(
        self,
        X: np.ndarray | None,
        labels: np.ndarray,
        snr: np.ndarray,
        class_names: list[str],
        dataset_path: str | Path | None = None,
        x_key: str = "X",
        sample_length: int = 1024,
    ):
        self.X = X
        self.labels = labels
        self.snr = snr
        self.class_names = class_names
        self.dataset_path = Path(dataset_path) if dataset_path is not None else None
        self.x_key = x_key
        self.sample_length = sample_length
        self.in_memory = X is not None
        self.num_classes = len(class_names)
        self.label_to_index = {name: i for i, name in enumerate(class_names)}
        self.index_to_label = {i: name for i, name in enumerate(class_names)}

    def __len__(self) -> int:
        return len(self.labels)

    def __repr__(self) -> str:
        return (
            f"RadioMLDataset(samples={len(self)}, classes={self.num_classes}, "
            f"snr_range=[{self.snr.min()}, {self.snr.max()}], "
            f"in_memory={self.in_memory})"
        )


def _validate_hdf5_keys(
    h5file: h5py.File,
    x_key: str,
    y_key: str,
    snr_key: str,
) -> None:
    """Validate that expected keys exist in the HDF5 file."""
    available = list(h5file.keys())
    for key, name in [(x_key, "I/Q data"), (y_key, "labels"), (snr_key, "SNR")]:
        if key not in available:
            raise KeyError(
                f"Expected HDF5 key '{key}' for {name} not found. "
                f"Available keys: {available}. "
                f"Check your config (x_key, y_key, snr_key) and dataset file."
            )


def _decode_labels(y_raw: np.ndarray) -> np.ndarray:
    """Decode labels from one-hot to integer indices.

    Parameters
    ----------
    y_raw : np.ndarray
        Raw label array, either one-hot (N, C) or integer (N,).

    Returns
    -------
    np.ndarray
        Integer labels, shape (N,).
    """
    if y_raw.ndim == 2 and y_raw.shape[1] > 1:
        logger.info(f"Decoding one-hot labels: shape {y_raw.shape}")
        return np.argmax(y_raw, axis=1).astype(np.int64)
    elif y_raw.ndim in (1, 2):
        return y_raw.reshape(-1).astype(np.int64)
    else:
        raise ValueError(f"Unexpected label shape: {y_raw.shape}")


def _decode_labels_dataset(y_dataset: h5py.Dataset, chunk_size: int = 100_000) -> np.ndarray:
    """Decode an HDF5 label dataset without keeping the one-hot matrix in memory."""
    n_samples = y_dataset.shape[0]
    if y_dataset.ndim == 2 and y_dataset.shape[1] > 1:
        labels = np.empty(n_samples, dtype=np.int64)
        logger.info(f"Decoding one-hot labels in chunks: shape {y_dataset.shape}")
        for start in range(0, n_samples, chunk_size):
            stop = min(start + chunk_size, n_samples)
            labels[start:stop] = np.argmax(y_dataset[start:stop], axis=1).astype(np.int64)
        return labels

    return np.asarray(y_dataset[:]).reshape(-1).astype(np.int64)


def _flatten_snr(snr_raw: np.ndarray) -> np.ndarray:
    """Flatten SNR array to 1D.

    Parameters
    ----------
    snr_raw : np.ndarray
        Raw SNR array, shape (N,) or (N, 1).

    Returns
    -------
    np.ndarray
        Flattened SNR values, shape (N,).
    """
    return snr_raw.flatten().astype(np.float32)


def _flatten_snr_dataset(snr_dataset: h5py.Dataset) -> np.ndarray:
    """Flatten an HDF5 SNR dataset to a 1D float32 array."""
    return np.asarray(snr_dataset[:]).reshape(-1).astype(np.float32)


def _resolve_class_names(num_classes: int, class_names: list[str] | None = None) -> list[str]:
    """Resolve class names, using RadioML defaults if available."""
    if class_names is not None:
        if len(class_names) != num_classes:
            raise ValueError(
                f"Configured class_names has {len(class_names)} entries, "
                f"but dataset labels require {num_classes} classes."
            )
        return class_names

    if num_classes == len(RADIOML_CLASS_NAMES):
        return RADIOML_CLASS_NAMES

    # Generate generic class names
    logger.warning(
        f"Could not match {num_classes} classes to known RadioML names. Using generic labels."
    )
    return [f"Class_{i}" for i in range(num_classes)]


def _load_class_names_from_config(data_cfg: dict[str, Any]) -> list[str] | None:
    """Load class names from config list or a JSON/text file if provided."""
    configured = data_cfg.get("class_names")
    if isinstance(configured, list):
        return [str(name) for name in configured]

    class_names_path = data_cfg.get("class_names_path")
    if not class_names_path:
        return None

    path = Path(class_names_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"Configured class_names_path not found: {path}")

    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            names = json.load(f)
    else:
        names = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    if not isinstance(names, list):
        raise ValueError(f"class_names_path must contain a list of class names: {path}")
    return [str(name) for name in names]


def _select_subset_indices(
    n_samples: int,
    max_samples: int | None,
    subset_fraction: float | None,
    seed: int = 42,
) -> np.ndarray | None:
    """Return sorted subset indices for development workflows, or None for all samples."""
    target = n_samples

    if subset_fraction is not None and 0 < subset_fraction < 1.0:
        target = max(1, int(n_samples * subset_fraction))
        logger.info(f"Applying subset_fraction={subset_fraction}: {n_samples} -> {target} samples")
    elif max_samples is not None and max_samples < n_samples:
        target = max(1, int(max_samples))
        logger.info(f"Applying max_samples={max_samples}: {n_samples} -> {target} samples")

    if target < n_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(n_samples, size=target, replace=False)
        indices.sort()
        return indices

    return None


def _estimate_array_gb(shape: tuple[int, ...], dtype: np.dtype) -> float:
    """Estimate an array size in GiB."""
    return float(np.prod(shape) * np.dtype(dtype).itemsize / (1024**3))


def load_radioml(
    cfg: dict[str, Any],
    class_names: list[str] | None = None,
) -> RadioMLDataset:
    """Load the RadioML 2018.01A dataset from HDF5.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary with 'data' section.
    class_names : list[str], optional
        Override class names. If None, uses RadioML defaults.

    Returns
    -------
    RadioMLDataset
        Loaded and validated dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    KeyError
        If expected HDF5 keys are missing.
    """
    data_cfg = cfg["data"]
    dataset_path = Path(data_cfg["dataset_path"])
    x_key = data_cfg.get("x_key", "X")
    y_key = data_cfg.get("y_key", "Y")
    snr_key = data_cfg.get("snr_key", "Z")
    max_samples = data_cfg.get("max_samples")
    subset_fraction = data_cfg.get("subset_fraction")
    seed = cfg.get("project", {}).get("seed", 42)

    # Validate file existence
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}\n"
            f"Please download the DeepSig RadioML 2018.01A dataset and place it at:\n"
            f"  {dataset_path}\n"
            f"Or update 'data.dataset_path' in your config file."
        )

    print_info(f"Loading dataset: {dataset_path}")
    print_info(f"HDF5 keys: X='{x_key}', Y='{y_key}', SNR='{snr_key}'")

    with h5py.File(str(dataset_path), "r") as f:
        _validate_hdf5_keys(f, x_key, y_key, snr_key)
        x_ds = f[x_key]
        y_ds = f[y_key]
        snr_ds = f[snr_key]

        print_info(f"  X shape: {x_ds.shape}, dtype: {x_ds.dtype}")
        print_info(f"  Y shape: {y_ds.shape}, dtype: {y_ds.dtype}")
        print_info(f"  Z shape: {snr_ds.shape}, dtype: {snr_ds.dtype}")

        if x_ds.ndim != 3 or x_ds.shape[2] != 2:
            raise ValueError(
                f"Expected I/Q data shape (N, sample_length, 2), got {x_ds.shape}. "
                "The last dimension should be 2 (I and Q channels)."
            )

        n_samples = x_ds.shape[0]
        expected_len = data_cfg.get("sample_length", 1024)
        actual_sample_length = x_ds.shape[1]
        if x_ds.shape[1] != expected_len:
            print_warning(
                f"Sample length {x_ds.shape[1]} differs from config sample_length={expected_len}. "
                "Using actual sample length from data."
            )

        if y_ds.shape[0] != n_samples or snr_ds.shape[0] != n_samples:
            raise ValueError(
                f"Sample count mismatch: X={n_samples}, Y={y_ds.shape[0]}, Z={snr_ds.shape[0]}"
            )

        print_info(f"Reading labels (key='{y_key}')...")
        labels = _decode_labels_dataset(y_ds)
        if len(labels) == 0:
            raise ValueError("Dataset contains zero labels.")
        num_classes = int(labels.max()) + 1

        print_info(f"Reading SNR (key='{snr_key}')...")
        snr = _flatten_snr_dataset(snr_ds)

        configured_names = class_names or _load_class_names_from_config(data_cfg)
        resolved_names = _resolve_class_names(num_classes, configured_names)

        subset_indices = _select_subset_indices(n_samples, max_samples, subset_fraction, seed)
        if subset_indices is not None:
            print_info(
                f"Reading I/Q subset (key='{x_key}'): {len(subset_indices)} of {n_samples} samples"
            )
            X = x_ds[subset_indices]
            labels = labels[subset_indices]
            snr = snr[subset_indices]
        else:
            estimated_gb = _estimate_array_gb(x_ds.shape, x_ds.dtype)
            if estimated_gb > 8:
                print_warning(
                    f"Full I/Q array is about {estimated_gb:.1f} GiB. Keeping X on disk and "
                    "using lazy HDF5 reads for training/evaluation."
                )
                X = None
            else:
                print_info(f"Reading full I/Q data (key='{x_key}')...")
                X = x_ds[:]

    if X is not None:
        X = X.astype(np.float32, copy=False)

    dataset = RadioMLDataset(
        X,
        labels,
        snr,
        resolved_names,
        dataset_path=dataset_path,
        x_key=x_key,
        sample_length=actual_sample_length,
    )
    print_info(f"Loaded {dataset}")

    return dataset
