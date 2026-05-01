"""Tests for waveform model forward passes."""

import torch

from rf_sentinel.models.cnn1d import CNN1D, build_cnn1d, count_parameters
from rf_sentinel.models.factory import build_waveform_model
from rf_sentinel.models.resnet1d import ResNet1D
from rf_sentinel.models.tcn1d import TCN1D


class TestCNN1DForward:
    def test_forward_default(self):
        model = CNN1D(input_channels=2, num_classes=24, base_channels=64)
        x = torch.randn(8, 2, 1024)
        out = model(x)
        assert out.shape == (8, 24)

    def test_forward_different_classes(self):
        model = CNN1D(input_channels=2, num_classes=10, base_channels=32)
        x = torch.randn(4, 2, 1024)
        out = model(x)
        assert out.shape == (4, 10)

    def test_forward_single_sample(self):
        model = CNN1D(input_channels=2, num_classes=24)
        x = torch.randn(1, 2, 1024)
        out = model(x)
        assert out.shape == (1, 24)

    def test_forward_different_length(self):
        model = CNN1D(input_channels=2, num_classes=24)
        x = torch.randn(4, 2, 512)
        out = model(x)
        assert out.shape == (4, 24)

    def test_output_not_nan(self):
        model = CNN1D(input_channels=2, num_classes=24)
        x = torch.randn(4, 2, 1024)
        out = model(x)
        assert not torch.any(torch.isnan(out))


class TestAdditionalWaveformModels:
    def test_resnet1d_forward(self):
        model = ResNet1D(
            input_channels=2,
            num_classes=24,
            base_channels=16,
            blocks_per_stage=[1, 1, 1],
        )
        x = torch.randn(4, 2, 1024)
        out = model(x)
        assert out.shape == (4, 24)
        assert not torch.any(torch.isnan(out))

    def test_tcn1d_forward(self):
        model = TCN1D(
            input_channels=2,
            num_classes=24,
            base_channels=16,
            num_blocks=3,
            kernel_size=5,
        )
        x = torch.randn(4, 2, 1024)
        out = model(x)
        assert out.shape == (4, 24)
        assert not torch.any(torch.isnan(out))


class TestModelUtils:
    def test_count_parameters(self):
        model = CNN1D(input_channels=2, num_classes=24, base_channels=64)
        n_params = count_parameters(model)
        assert n_params > 0
        assert isinstance(n_params, int)

    def test_build_from_config(self):
        cfg = {
            "model": {"input_channels": 2, "num_classes": 10, "base_channels": 32, "dropout": 0.5}
        }
        model = build_cnn1d(cfg)
        assert model.num_classes == 10
        assert model.base_channels == 32

    def test_build_resnet_from_factory(self):
        cfg = {
            "model": {
                "type": "resnet1d",
                "input_channels": 2,
                "num_classes": 10,
                "base_channels": 16,
                "blocks_per_stage": [1, 1, 1],
            }
        }
        model = build_waveform_model(cfg)
        assert model.num_classes == 10
        assert model.base_channels == 16

    def test_build_tcn_from_factory(self):
        cfg = {
            "model": {
                "type": "tcn1d",
                "input_channels": 2,
                "num_classes": 10,
                "base_channels": 16,
                "num_blocks": 3,
            }
        }
        model = build_waveform_model(cfg)
        assert model.num_classes == 10
        assert model.base_channels == 16
