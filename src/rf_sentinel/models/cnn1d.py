"""CNN1D model for RF modulation classification from raw I/Q data."""

from __future__ import annotations

import torch
import torch.nn as nn

from rf_sentinel.utils.logging import print_info


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for modulation classification.

    Architecture:
        Conv1d(2 -> base) -> BN -> ReLU -> MaxPool
        Conv1d(base -> base*2) -> BN -> ReLU -> MaxPool
        Conv1d(base*2 -> base*4) -> BN -> ReLU -> AdaptiveAvgPool
        Dropout -> Linear -> num_classes

    Parameters
    ----------
    input_channels : int
        Number of input channels (2 for I/Q).
    num_classes : int
        Number of output modulation classes.
    base_channels : int
        Number of channels in the first conv layer.
    dropout : float
        Dropout probability before the final linear layer.
    """

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 24,
        base_channels: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 2
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 3
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, 2, sample_length).

        Returns
        -------
        torch.Tensor
            Logits, shape (batch, num_classes).
        """
        x = self.features(x)
        x = x.squeeze(-1)  # (batch, channels, 1) -> (batch, channels)
        x = self.classifier(x)
        return x


def build_cnn1d(cfg: dict) -> CNN1D:
    """Build CNN1D model from configuration."""
    model_cfg = cfg.get("model", {})
    num_classes = model_cfg.get("num_classes", 24)
    if num_classes == "auto":
        num_classes = 24  # RadioML default, will be overridden at runtime

    model = CNN1D(
        input_channels=model_cfg.get("input_channels", 2),
        num_classes=num_classes,
        base_channels=model_cfg.get("base_channels", 64),
        dropout=model_cfg.get("dropout", 0.25),
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: CNN1D) -> None:
    """Print model architecture summary."""
    n_params = count_parameters(model)
    print_info("Model: CNN1D")
    print_info(f"  Input channels: {model.input_channels}")
    print_info(f"  Output classes: {model.num_classes}")
    print_info(f"  Base channels: {model.base_channels}")
    print_info(f"  Trainable parameters: {n_params:,}")
