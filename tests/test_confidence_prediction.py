"""Tests for confidence-aware prediction."""

import numpy as np
import pytest
import torch

from rf_sentinel.models.cnn1d import CNN1D
from rf_sentinel.models.resnet1d import ResNet1D
from rf_sentinel.models.tcn1d import TCN1D


def _write_dummy_checkpoint(tmp_path, model, model_cfg, filename):
    """Create a dummy waveform-model checkpoint for testing."""
    class_names = [f"Mod_{i}" for i in range(8)]

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {"model": model_cfg},
        "model_config": model_cfg,
        "class_names": class_names,
        "label_to_index": {name: i for i, name in enumerate(class_names)},
        "index_to_label": {i: name for i, name in enumerate(class_names)},
        "normalization": "rms",
        "sample_length": 1024,
        "model_type": model_cfg["type"],
        "input_channels": 2,
        "num_classes": 8,
        "base_channels": model_cfg.get("base_channels", 32),
        "dropout": model_cfg.get("dropout", 0.25),
    }

    path = tmp_path / filename
    torch.save(checkpoint, str(path))
    return path


@pytest.fixture
def dummy_checkpoint(tmp_path):
    """Create a dummy CNN1D checkpoint for testing."""
    model_cfg = {"type": "cnn1d", "input_channels": 2, "num_classes": 8, "base_channels": 32}
    model = CNN1D(input_channels=2, num_classes=8, base_channels=32)
    return _write_dummy_checkpoint(tmp_path, model, model_cfg, "test_cnn1d_checkpoint.pt")


@pytest.fixture
def resnet_checkpoint(tmp_path):
    """Create a dummy ResNet1D checkpoint for testing."""
    model_cfg = {
        "type": "resnet1d",
        "input_channels": 2,
        "num_classes": 8,
        "base_channels": 8,
        "blocks_per_stage": [1, 1, 1],
    }
    model = ResNet1D(
        input_channels=2,
        num_classes=8,
        base_channels=8,
        blocks_per_stage=[1, 1, 1],
    )
    return _write_dummy_checkpoint(tmp_path, model, model_cfg, "test_resnet1d_checkpoint.pt")


@pytest.fixture
def tcn_checkpoint(tmp_path):
    """Create a dummy TCN1D checkpoint for testing."""
    model_cfg = {
        "type": "tcn1d",
        "input_channels": 2,
        "num_classes": 8,
        "base_channels": 8,
        "num_blocks": 2,
        "kernel_size": 5,
    }
    model = TCN1D(
        input_channels=2,
        num_classes=8,
        base_channels=8,
        num_blocks=2,
        kernel_size=5,
    )
    return _write_dummy_checkpoint(tmp_path, model, model_cfg, "test_tcn1d_checkpoint.pt")


class TestConfidencePrediction:
    def test_predict_returns_expected_keys(self, dummy_checkpoint):
        from rf_sentinel.inference.predictor import RFPredictor

        predictor = RFPredictor(
            checkpoint_path=dummy_checkpoint,
            device_cfg="cpu",
            confidence_threshold=0.70,
            top_k=3,
        )

        sample = np.random.randn(1024, 2).astype(np.float32)
        result = predictor.predict(sample)

        assert "prediction" in result
        assert "confidence" in result
        assert "decision" in result
        assert "top_k" in result

    def test_confidence_is_valid(self, dummy_checkpoint):
        from rf_sentinel.inference.predictor import RFPredictor

        predictor = RFPredictor(
            checkpoint_path=dummy_checkpoint,
            device_cfg="cpu",
        )

        sample = np.random.randn(1024, 2).astype(np.float32)
        result = predictor.predict(sample)

        assert 0 <= result["confidence"] <= 1.0

    def test_decision_logic(self, dummy_checkpoint):
        from rf_sentinel.inference.predictor import RFPredictor

        predictor = RFPredictor(
            checkpoint_path=dummy_checkpoint,
            device_cfg="cpu",
            confidence_threshold=0.0,  # Everything should be accepted
        )

        sample = np.random.randn(1024, 2).astype(np.float32)
        result = predictor.predict(sample)
        assert result["decision"] == "accepted"

    def test_top_k_count(self, dummy_checkpoint):
        from rf_sentinel.inference.predictor import RFPredictor

        predictor = RFPredictor(
            checkpoint_path=dummy_checkpoint,
            device_cfg="cpu",
            top_k=5,
        )

        sample = np.random.randn(1024, 2).astype(np.float32)
        result = predictor.predict(sample)
        assert len(result["top_k"]) == 5

    def test_batch_predict(self, dummy_checkpoint):
        from rf_sentinel.inference.predictor import RFPredictor

        predictor = RFPredictor(
            checkpoint_path=dummy_checkpoint,
            device_cfg="cpu",
        )

        X = np.random.randn(20, 2, 1024).astype(np.float32)
        preds, probs = predictor.predict_batch(X)

        assert preds.shape == (20,)
        assert probs.shape == (20, 8)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_wrong_sample_length_raises(self, dummy_checkpoint):
        from rf_sentinel.inference.predictor import RFPredictor

        predictor = RFPredictor(
            checkpoint_path=dummy_checkpoint,
            device_cfg="cpu",
        )

        sample = np.random.randn(512, 2).astype(np.float32)
        with pytest.raises(ValueError, match="Expected sample length"):
            predictor.predict(sample)

    def test_wrong_batch_length_raises(self, dummy_checkpoint):
        from rf_sentinel.inference.predictor import RFPredictor

        predictor = RFPredictor(
            checkpoint_path=dummy_checkpoint,
            device_cfg="cpu",
        )

        X = np.random.randn(4, 2, 512).astype(np.float32)
        with pytest.raises(ValueError, match="Expected sample length"):
            predictor.predict_batch(X)

    def test_predict_resnet_checkpoint(self, resnet_checkpoint):
        from rf_sentinel.inference.predictor import RFPredictor

        predictor = RFPredictor(checkpoint_path=resnet_checkpoint, device_cfg="cpu")
        sample = np.random.randn(1024, 2).astype(np.float32)
        result = predictor.predict(sample)
        assert "prediction" in result
        assert result["decision"] in {"accepted", "uncertain"}

    def test_predict_tcn_checkpoint(self, tcn_checkpoint):
        from rf_sentinel.inference.predictor import RFPredictor

        predictor = RFPredictor(checkpoint_path=tcn_checkpoint, device_cfg="cpu")
        sample = np.random.randn(1024, 2).astype(np.float32)
        result = predictor.predict(sample)
        assert "prediction" in result
        assert result["decision"] in {"accepted", "uncertain"}
