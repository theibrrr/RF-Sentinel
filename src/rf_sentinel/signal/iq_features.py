"""I/Q feature extraction for traditional ML baselines (e.g., XGBoost)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

try:
    from scipy import stats as sp_stats

    HAS_SCIPY_STATS = True
except ImportError:
    HAS_SCIPY_STATS = False


def amplitude_stats(i_data: np.ndarray, q_data: np.ndarray) -> dict[str, float]:
    """Amplitude statistics from I/Q."""
    amp = np.sqrt(i_data**2 + q_data**2)
    return {
        "amp_mean": float(np.mean(amp)),
        "amp_std": float(np.std(amp)),
        "amp_min": float(np.min(amp)),
        "amp_max": float(np.max(amp)),
        "amp_median": float(np.median(amp)),
    }


def phase_stats(i_data: np.ndarray, q_data: np.ndarray) -> dict[str, float]:
    """Phase statistics from I/Q."""
    phase = np.arctan2(q_data, i_data)
    return {
        "phase_mean": float(np.mean(phase)),
        "phase_std": float(np.std(phase)),
    }


def instantaneous_frequency(i_data: np.ndarray, q_data: np.ndarray) -> dict[str, float]:
    """Instantaneous frequency proxy from unwrapped phase difference."""
    phase = np.arctan2(q_data, i_data)
    unwrapped = np.unwrap(phase)
    inst_freq = np.diff(unwrapped)
    return {
        "inst_freq_mean": float(np.mean(inst_freq)),
        "inst_freq_std": float(np.std(inst_freq)),
    }


def energy_features(i_data: np.ndarray, q_data: np.ndarray) -> dict[str, float]:
    """Signal energy features."""
    energy = np.sum(i_data**2 + q_data**2)
    return {"energy": float(energy)}


def iq_stats(i_data: np.ndarray, q_data: np.ndarray) -> dict[str, float]:
    """Basic I and Q channel statistics."""
    return {
        "i_mean": float(np.mean(i_data)),
        "i_std": float(np.std(i_data)),
        "q_mean": float(np.mean(q_data)),
        "q_std": float(np.std(q_data)),
    }


def higher_order_stats(i_data: np.ndarray, q_data: np.ndarray) -> dict[str, float]:
    """Higher-order statistics (kurtosis, skewness) of amplitude."""
    if not HAS_SCIPY_STATS:
        return {}
    amp = np.sqrt(i_data**2 + q_data**2)
    return {
        "amp_kurtosis": float(sp_stats.kurtosis(amp)),
        "amp_skewness": float(sp_stats.skew(amp)),
    }


def spectral_entropy(i_data: np.ndarray, q_data: np.ndarray) -> dict[str, float]:
    """Spectral entropy of the complex signal."""
    sig = i_data + 1j * q_data
    spectrum = np.abs(np.fft.fft(sig))
    spectrum = spectrum / (np.sum(spectrum) + 1e-10)
    spectrum = spectrum[spectrum > 0]
    entropy = -np.sum(spectrum * np.log2(spectrum + 1e-10))
    return {"spectral_entropy": float(entropy)}


# Registry of feature extractors
FEATURE_REGISTRY: dict[str, Callable[[np.ndarray, np.ndarray], dict[str, float]]] = {
    "amplitude_stats": amplitude_stats,
    "phase_stats": phase_stats,
    "instantaneous_frequency": instantaneous_frequency,
    "energy": energy_features,
    "iq_stats": iq_stats,
    "higher_order_stats": higher_order_stats,
    "spectral_entropy": spectral_entropy,
}


def extract_features(
    sample: np.ndarray,
    feature_set: list[str] | None = None,
) -> dict[str, float]:
    """Extract features from a single I/Q sample.

    Parameters
    ----------
    sample : np.ndarray
        Single I/Q sample, shape (L, 2) or (2, L).
    feature_set : list[str], optional
        List of feature group names. If None, extracts all.

    Returns
    -------
    dict[str, float]
        Extracted feature values.
    """
    if sample.ndim != 2:
        raise ValueError(f"Expected 2D sample, got shape {sample.shape}")

    if sample.shape[0] == 2 and sample.shape[1] > 2:
        sample = sample.T  # (2, L) -> (L, 2)

    i_data, q_data = sample[:, 0], sample[:, 1]

    if feature_set is None:
        feature_set = list(FEATURE_REGISTRY.keys())

    features = {}
    for name in feature_set:
        if name in FEATURE_REGISTRY:
            features.update(FEATURE_REGISTRY[name](i_data, q_data))

    return features


def extract_features_batch(
    X: np.ndarray,
    feature_set: list[str] | None = None,
) -> np.ndarray:
    """Extract features from a batch of I/Q samples.

    Parameters
    ----------
    X : np.ndarray
        Batch of I/Q samples, shape (N, L, 2).
    feature_set : list[str], optional
        Feature group names.

    Returns
    -------
    np.ndarray
        Feature matrix, shape (N, num_features).
    """
    all_features = []
    for i in range(len(X)):
        feat_dict = extract_features(X[i], feature_set)
        all_features.append(list(feat_dict.values()))

    return np.array(all_features, dtype=np.float32)


def get_feature_names(feature_set: list[str] | None = None) -> list[str]:
    """Get ordered feature names for a given feature set."""
    dummy = np.zeros((10, 2), dtype=np.float32)
    feat_dict = extract_features(dummy, feature_set)
    return list(feat_dict.keys())
