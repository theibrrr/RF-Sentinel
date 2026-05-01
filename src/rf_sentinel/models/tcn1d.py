"""Dilated TCN-style 1D model for RF modulation classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBlock1D(nn.Module):
    """Non-causal dilated temporal block with a residual path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("TemporalBlock1D requires an odd kernel_size for same-length padding.")

        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.shortcut(x))


class TCN1D(nn.Module):
    """Dilated convolutional temporal network for raw I/Q waveforms.

    The network expands its receptive field through exponentially increasing
    dilations while preserving the same input/output interface as the CNN1D
    baseline: ``(batch, 2, sample_length)`` -> ``(batch, num_classes)``.
    """

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 24,
        base_channels: int = 64,
        num_blocks: int = 5,
        kernel_size: int = 5,
        dilation_base: int = 2,
        dropout: float = 0.20,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("TCN1D requires at least one temporal block.")

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.dropout = dropout

        layers = []
        in_channels = input_channels
        for block_idx in range(num_blocks):
            out_channels = base_channels * min(2 ** (block_idx // 2), 4)
            dilation = dilation_base**block_idx
            layers.append(
                TemporalBlock1D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def build_tcn1d(cfg: dict) -> TCN1D:
    """Build a TCN1D model from configuration."""
    model_cfg = cfg.get("model", {})
    num_classes = model_cfg.get("num_classes", 24)
    if num_classes == "auto":
        num_classes = 24

    return TCN1D(
        input_channels=model_cfg.get("input_channels", 2),
        num_classes=num_classes,
        base_channels=model_cfg.get("base_channels", 64),
        num_blocks=model_cfg.get("num_blocks", 5),
        kernel_size=model_cfg.get("kernel_size", 5),
        dilation_base=model_cfg.get("dilation_base", 2),
        dropout=model_cfg.get("dropout", 0.20),
    )
