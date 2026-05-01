"""Tests for configuration loading."""

import pytest
import yaml

from rf_sentinel.config.loader import load_config, validate_config


@pytest.fixture
def minimal_config(tmp_path):
    """Create a minimal valid config file."""
    cfg = {
        "project": {"name": "test", "seed": 42},
        "data": {"dataset_path": "test.hdf5"},
        "model": {"type": "cnn1d"},
        "splits": {"train_size": 0.7, "val_size": 0.15, "test_size": 0.15},
    }
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


def test_load_config_basic(minimal_config):
    cfg = load_config(minimal_config)
    assert cfg["project"]["name"] == "test"
    assert cfg["project"]["seed"] == 42
    assert cfg["model"]["type"] == "cnn1d"


def test_load_config_defaults(minimal_config):
    cfg = load_config(minimal_config)
    # Check that defaults are applied
    assert "training" in cfg
    assert cfg["training"]["epochs"] == 30
    assert cfg["training"]["batch_size"] == 256


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")


def test_validate_config_bad_splits():
    cfg = {
        "project": {"name": "test"},
        "data": {"dataset_path": "test.hdf5"},
        "model": {"type": "cnn1d"},
        "splits": {"train_size": 0.5, "val_size": 0.5, "test_size": 0.5},
    }
    warnings = validate_config(cfg)
    assert any("sum" in w.lower() for w in warnings)


def test_validate_config_negative_lr():
    cfg = {
        "project": {"name": "test"},
        "data": {"dataset_path": "test.hdf5"},
        "model": {"type": "cnn1d"},
        "training": {"learning_rate": -0.01},
    }
    warnings = validate_config(cfg)
    assert any("learning rate" in w.lower() for w in warnings)
