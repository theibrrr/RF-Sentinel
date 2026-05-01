"""Device selection utilities for PyTorch."""

from __future__ import annotations

import torch

from rf_sentinel.utils.logging import get_logger

logger = get_logger(__name__)


def get_device(device_cfg: str = "auto") -> torch.device:
    """Select compute device based on configuration.

    Parameters
    ----------
    device_cfg : str
        Device specification: "auto", "cpu", "cuda", or "cuda:N".

    Returns
    -------
    torch.device
        Selected PyTorch device.
    """
    if device_cfg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA device: {gpu_name}")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")
    else:
        device = torch.device(device_cfg)
        logger.info(f"Using device: {device}")

    return device
