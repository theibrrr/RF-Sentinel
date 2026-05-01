"""End-to-end smoke test using synthetic data (no real dataset required)."""

import numpy as np
import pytest
import torch

from rf_sentinel.data.preprocessing import prepare_single_sample, preprocess_dataset
from rf_sentinel.models.cnn1d import CNN1D


class TestSyntheticSmokeTest:
    """Full pipeline smoke test on synthetic data."""

    @pytest.fixture
    def synthetic_data(self):
        np.random.seed(42)
        n_samples = 100
        sample_length = 1024
        num_classes = 8
        X = np.random.randn(n_samples, sample_length, 2).astype(np.float32)
        labels = np.random.randint(0, num_classes, size=n_samples)
        snr = np.random.choice(np.arange(-20, 32, 2), size=n_samples).astype(np.float32)
        return X, labels, snr, num_classes

    def test_preprocessing_pipeline(self, synthetic_data):
        X, _, _, _ = synthetic_data
        X_proc = preprocess_dataset(X, normalize_method="rms")
        assert X_proc.shape == (100, 2, 1024)
        assert X_proc.dtype == np.float32
        assert np.all(np.isfinite(X_proc))

    def test_model_forward_pass(self, synthetic_data):
        X, _, _, num_classes = synthetic_data
        X_proc = preprocess_dataset(X)
        model = CNN1D(input_channels=2, num_classes=num_classes)

        x_batch = torch.from_numpy(X_proc[:8]).float()
        with torch.no_grad():
            out = model(x_batch)

        assert out.shape == (8, num_classes)
        assert not torch.any(torch.isnan(out))

    def test_mini_training_loop(self, synthetic_data):
        X, labels, _, num_classes = synthetic_data
        X_proc = preprocess_dataset(X)

        model = CNN1D(input_channels=2, num_classes=num_classes, base_channels=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        x_tensor = torch.from_numpy(X_proc[:32]).float()
        y_tensor = torch.from_numpy(labels[:32]).long()

        optimizer.zero_grad()
        logits = model(x_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert np.isfinite(loss.item())

    def test_prediction_pipeline(self, synthetic_data):
        X, _, _, num_classes = synthetic_data
        X_proc = preprocess_dataset(X)

        model = CNN1D(input_channels=2, num_classes=num_classes)
        model.eval()

        with torch.no_grad():
            x = torch.from_numpy(X_proc[:1]).float()
            logits = model(x)
            probs = torch.nn.functional.softmax(logits, dim=1)

        assert probs.shape == (1, num_classes)
        assert abs(probs.sum().item() - 1.0) < 1e-5

        pred = probs.argmax(dim=1).item()
        conf = probs.max().item()
        assert 0 <= pred < num_classes
        assert 0 <= conf <= 1.0

    def test_single_sample_inference(self, synthetic_data):
        X, _, _, _ = synthetic_data
        sample = X[0]  # (1024, 2)
        prepared = prepare_single_sample(sample, normalize_method="rms")
        assert prepared.shape == (1, 2, 1024)

    def test_feature_extraction(self, synthetic_data):
        X, _, _, _ = synthetic_data
        from rf_sentinel.signal.iq_features import extract_features, extract_features_batch

        feats = extract_features(X[0])
        assert len(feats) > 0
        assert all(isinstance(v, float) for v in feats.values())

        batch_feats = extract_features_batch(X[:5])
        assert batch_feats.shape[0] == 5
        assert batch_feats.shape[1] == len(feats)
