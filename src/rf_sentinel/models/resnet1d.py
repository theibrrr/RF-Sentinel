"""ResNet1D model for RF modulation classification from raw I/Q data."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Basic residual block for 1D I/Q waveform features."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class ResNet1D(nn.Module):
    """Compact 1D residual network for raw I/Q modulation classification.

    Input shape is ``(batch, 2, sample_length)`` and output shape is
    ``(batch, num_classes)``. The model keeps the same raw waveform interface as
    ``CNN1D`` while using residual blocks for deeper feature learning.
    """

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 24,
        base_channels: int = 64,
        dropout: float = 0.20,
        blocks_per_stage: list[int] | tuple[int, int, int] = (2, 2, 2),
    ):
        super().__init__()
        if len(blocks_per_stage) != 3:
            raise ValueError("ResNet1D expects exactly three stage block counts.")

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.dropout = dropout
        self.blocks_per_stage = list(blocks_per_stage)

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.in_channels = base_channels
        self.layer1 = self._make_stage(base_channels, blocks_per_stage[0], stride=1)
        self.layer2 = self._make_stage(base_channels * 2, blocks_per_stage[1], stride=2)
        self.layer3 = self._make_stage(base_channels * 4, blocks_per_stage[2], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_channels * 4, num_classes),
        )

    def _make_stage(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock1D(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def build_resnet1d(cfg: dict) -> ResNet1D:
    """Build a ResNet1D model from configuration."""
    model_cfg = cfg.get("model", {})
    num_classes = model_cfg.get("num_classes", 24)
    if num_classes == "auto":
        num_classes = 24

    return ResNet1D(
        input_channels=model_cfg.get("input_channels", 2),
        num_classes=num_classes,
        base_channels=model_cfg.get("base_channels", 64),
        dropout=model_cfg.get("dropout", 0.20),
        blocks_per_stage=model_cfg.get("blocks_per_stage", [2, 2, 2]),
    )
