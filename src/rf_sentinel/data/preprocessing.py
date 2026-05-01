"""I/Q data preprocessing — normalization and shape conversion."""

from __future__ import annotations

import numpy as np

from rf_sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# Small constant for numerical stability in normalization
_EPS = 1e-10


def rms_normalize(x: np.ndarray) -> np.ndarray:
    """Per-sample RMS normalization of I/Q data.

    For each sample, compute RMS = sqrt(mean(I^2 + Q^2)) and divide by RMS.

    Parameters
    ----------
    x : np.ndarray
        I/Q data, shape (N, sample_length, 2) or (sample_length, 2).

    Returns
    -------
    np.ndarray
        RMS-normalized data, same shape as input.
    """
    single = x.ndim == 2
    if single:
        x = x[np.newaxis, ...]

    # x shape: (N, L, 2)
    power = x[:, :, 0] ** 2 + x[:, :, 1] ** 2  # (N, L)
    rms = np.sqrt(np.mean(power, axis=1, keepdims=True))  # (N, 1)
    rms = np.maximum(rms, _EPS)
    x_norm = x / rms[:, :, np.newaxis]  # broadcast (N, 1, 1)

    if single:
        return x_norm[0]
    return x_norm


def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """Per-sample z-score normalization of I/Q data.

    Each channel (I and Q) is normalized independently per sample.

    Parameters
    ----------
    x : np.ndarray
        I/Q data, shape (N, sample_length, 2) or (sample_length, 2).

    Returns
    -------
    np.ndarray
        Z-score normalized data.
    """
    single = x.ndim == 2
    if single:
        x = x[np.newaxis, ...]

    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    std = np.maximum(std, _EPS)
    x_norm = (x - mean) / std

    if single:
        return x_norm[0]
    return x_norm


def _validate_channels_last_iq(x: np.ndarray) -> None:
    """Validate I/Q data in channels-last format."""
    if x.ndim == 2 and x.shape[1] == 2:
        return
    if x.ndim == 3 and x.shape[2] == 2:
        return
    raise ValueError(f"Expected I/Q data as (L, 2) or (N, L, 2), got shape {x.shape}.")


def normalize_iq(x: np.ndarray, method: str = "rms") -> np.ndarray:
    """Apply normalization to I/Q data.

    Parameters
    ----------
    x : np.ndarray
        I/Q data, shape (N, sample_length, 2) or (sample_length, 2).
    method : str
        Normalization method: "rms", "zscore", or "none".

    Returns
    -------
    np.ndarray
        Normalized data.
    """
    _validate_channels_last_iq(x)

    if method == "rms":
        return rms_normalize(x)
    elif method == "zscore":
        return zscore_normalize(x)
    elif method == "none" or method is None:
        return x.copy()
    else:
        raise ValueError(
            f"Unknown normalization method: '{method}'. Use 'rms', 'zscore', or 'none'."
        )


def convert_to_channels_first(x: np.ndarray) -> np.ndarray:
    """Convert I/Q data from (N, L, 2) to (N, 2, L) for PyTorch Conv1d.

    Parameters
    ----------
    x : np.ndarray
        I/Q data, shape (N, sample_length, 2).

    Returns
    -------
    np.ndarray
        Transposed data, shape (N, 2, sample_length).
    """
    if x.ndim == 3 and x.shape[2] == 2:
        return np.transpose(x, (0, 2, 1)).astype(np.float32)
    elif x.ndim == 3 and x.shape[1] == 2:
        # Already channels-first
        return x.astype(np.float32)
    else:
        raise ValueError(f"Expected shape (N, L, 2) or (N, 2, L), got {x.shape}")


def prepare_single_sample(
    x: np.ndarray,
    normalize_method: str = "rms",
    expected_length: int | None = None,
) -> np.ndarray:
    """Prepare a single I/Q sample for inference.

    Accepts (L, 2) or (2, L) and returns (1, 2, L) normalized.

    Parameters
    ----------
    x : np.ndarray
        Single sample, shape (sample_length, 2) or (2, sample_length).
    normalize_method : str
        Normalization method.
    expected_length : int, optional
        Required sample length. If provided, mismatched samples are rejected.

    Returns
    -------
    np.ndarray
        Prepared sample, shape (1, 2, sample_length).
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D single sample (L, 2) or (2, L), got shape {x.shape}")

    # Detect orientation: if shape is (2, L) with L >> 2, it's channels-first
    if x.shape[0] == 2 and x.shape[1] > 2:
        # Convert (2, L) -> (L, 2) for normalization
        x = x.T

    if x.shape[1] != 2:
        raise ValueError(
            f"Expected 2 channels (I/Q), got shape {x.shape}. "
            f"Provide data as (sample_length, 2) or (2, sample_length)."
        )

    if expected_length is not None and x.shape[0] != expected_length:
        raise ValueError(
            f"Expected sample length {expected_length}, got {x.shape[0]}. "
            f"Provide a sample shaped ({expected_length}, 2) or (2, {expected_length})."
        )

    # Normalize in (L, 2) format
    x = normalize_iq(x, method=normalize_method)

    # Convert to (1, 2, L)
    x = x.T[np.newaxis, ...]  # (L, 2) -> (2, L) -> (1, 2, L)

    return x.astype(np.float32)


def preprocess_dataset(
    X: np.ndarray,
    normalize_method: str = "rms",
) -> np.ndarray:
    """Full preprocessing pipeline for dataset I/Q data.

    Parameters
    ----------
    X : np.ndarray
        Raw I/Q data, shape (N, sample_length, 2).
    normalize_method : str
        Normalization method.

    Returns
    -------
    np.ndarray
        Preprocessed data, shape (N, 2, sample_length), float32.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array (N, L, 2) or (N, 2, L), got shape {X.shape}")

    if X.shape[2] == 2:
        X_channels_last = X
    elif X.shape[1] == 2 and X.shape[2] > 2:
        X_channels_last = np.transpose(X, (0, 2, 1))
    else:
        raise ValueError(f"Expected I/Q data shape (N, L, 2) or (N, 2, L), got {X.shape}.")

    # Ensure float32
    X_channels_last = X_channels_last.astype(np.float32, copy=False)

    # Normalize in (N, L, 2) format
    X_channels_last = normalize_iq(X_channels_last, method=normalize_method)

    # Convert to channels-first (N, 2, L)
    X_processed = convert_to_channels_first(X_channels_last)

    logger.info(f"Preprocessed data: shape={X_processed.shape}, dtype={X_processed.dtype}")
    return X_processed
