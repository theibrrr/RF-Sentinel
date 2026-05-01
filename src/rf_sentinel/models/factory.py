"""Model factory for RF-Sentinel waveform classifiers."""

from __future__ import annotations

import torch.nn as nn

from rf_sentinel.models.cnn1d import build_cnn1d
from rf_sentinel.models.resnet1d import build_resnet1d
from rf_sentinel.models.tcn1d import build_tcn1d
from rf_sentinel.utils.logging import print_info

WAVEFORM_MODEL_TYPES = ("cnn1d", "resnet1d", "tcn1d")


def get_model_type(cfg: dict) -> str:
    """Return normalized model type from config."""
    return str(cfg.get("model", {}).get("type", "cnn1d")).lower()


def build_waveform_model(cfg: dict) -> nn.Module:
    """Build a raw-I/Q waveform model selected by ``model.type``."""
    model_type = get_model_type(cfg)
    if model_type == "cnn1d":
        return build_cnn1d(cfg)
    if model_type == "resnet1d":
        return build_resnet1d(cfg)
    if model_type == "tcn1d":
        return build_tcn1d(cfg)

    supported = ", ".join(WAVEFORM_MODEL_TYPES)
    raise ValueError(
        f"Unsupported waveform model type '{model_type}'. "
        f"Use one of: {supported}. Use train-xgboost for the XGBoost baseline."
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def checkpoint_stem(model_type: str) -> str:
    """Return the checkpoint filename stem for a model type."""
    normalized = model_type.lower()
    if normalized not in WAVEFORM_MODEL_TYPES:
        raise ValueError(f"Unsupported checkpoint model type: {model_type}")
    return normalized


def print_model_summary(model: nn.Module, model_type: str) -> None:
    """Print a compact model summary for any supported waveform model."""
    print_info(f"Model: {model_type}")
    print_info(f"  Input channels: {getattr(model, 'input_channels', 'unknown')}")
    print_info(f"  Output classes: {getattr(model, 'num_classes', 'unknown')}")
    print_info(f"  Base channels: {getattr(model, 'base_channels', 'unknown')}")

    if hasattr(model, "blocks_per_stage"):
        print_info(f"  Blocks per stage: {model.blocks_per_stage}")
    if hasattr(model, "num_blocks"):
        print_info(f"  Temporal blocks: {model.num_blocks}")
        print_info(f"  Kernel size: {model.kernel_size}")
        print_info(f"  Dilation base: {model.dilation_base}")

    print_info(f"  Trainable parameters: {count_parameters(model):,}")
