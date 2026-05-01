"""Tests for I/Q preprocessing."""

import numpy as np
import pytest

from rf_sentinel.data.preprocessing import (
    convert_to_channels_first,
    prepare_single_sample,
    preprocess_dataset,
    rms_normalize,
    zscore_normalize,
)


class TestRMSNormalization:
    def test_rms_normalize_batch(self):
        X = np.random.randn(10, 1024, 2).astype(np.float32)
        X_norm = rms_normalize(X)
        assert X_norm.shape == X.shape
        assert X_norm.dtype == X.dtype

    def test_rms_normalize_single(self):
        x = np.random.randn(1024, 2).astype(np.float32)
        x_norm = rms_normalize(x)
        assert x_norm.shape == x.shape

    def test_rms_normalize_zero_signal(self):
        x = np.zeros((1024, 2), dtype=np.float32)
        x_norm = rms_normalize(x)
        assert np.all(np.isfinite(x_norm))

    def test_rms_normalize_changes_scale(self):
        x = np.random.randn(1024, 2).astype(np.float32) * 100
        x_norm = rms_normalize(x)
        rms_after = np.sqrt(np.mean(x_norm[:, 0] ** 2 + x_norm[:, 1] ** 2))
        assert abs(rms_after - 1.0) < 0.1


class TestZScoreNormalization:
    def test_zscore_batch(self):
        X = np.random.randn(5, 1024, 2).astype(np.float32) * 10 + 5
        X_norm = zscore_normalize(X)
        assert X_norm.shape == X.shape

    def test_zscore_zero_variance(self):
        x = np.ones((1024, 2), dtype=np.float32)
        x_norm = zscore_normalize(x)
        assert np.all(np.isfinite(x_norm))


class TestShapeConversion:
    def test_channels_first(self):
        X = np.random.randn(10, 1024, 2).astype(np.float32)
        X_cf = convert_to_channels_first(X)
        assert X_cf.shape == (10, 2, 1024)

    def test_already_channels_first(self):
        X = np.random.randn(10, 2, 1024).astype(np.float32)
        X_cf = convert_to_channels_first(X)
        assert X_cf.shape == (10, 2, 1024)


class TestPreprocessDataset:
    def test_full_pipeline(self):
        X = np.random.randn(20, 1024, 2).astype(np.float32)
        result = preprocess_dataset(X, normalize_method="rms")
        assert result.shape == (20, 2, 1024)
        assert result.dtype == np.float32

    def test_no_normalization(self):
        X = np.random.randn(5, 1024, 2).astype(np.float32)
        result = preprocess_dataset(X, normalize_method="none")
        assert result.shape == (5, 2, 1024)

    def test_channels_first_pipeline_normalizes_correct_axis(self):
        X = np.random.randn(5, 2, 1024).astype(np.float32) * 5
        result = preprocess_dataset(X, normalize_method="rms")
        rms = np.sqrt(np.mean(result[:, 0, :] ** 2 + result[:, 1, :] ** 2, axis=1))
        assert result.shape == (5, 2, 1024)
        assert np.allclose(rms, 1.0, atol=1e-5)


class TestSingleSamplePrep:
    def test_channels_last(self):
        x = np.random.randn(1024, 2).astype(np.float32)
        result = prepare_single_sample(x, normalize_method="rms")
        assert result.shape == (1, 2, 1024)

    def test_channels_first(self):
        x = np.random.randn(2, 1024).astype(np.float32)
        result = prepare_single_sample(x, normalize_method="rms")
        assert result.shape == (1, 2, 1024)

    def test_wrong_shape_raises(self):
        x = np.random.randn(3, 1024).astype(np.float32)
        with pytest.raises(ValueError):
            prepare_single_sample(x)
