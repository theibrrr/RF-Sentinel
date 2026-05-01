"""Reproducibility utilities — seed setting for deterministic experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from rf_sentinel.utils.logging import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, PyTorch, and CUDA.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Global seed set to {seed}")


def get_seed_from_config(cfg: dict) -> int:
    """Extract seed from configuration dictionary."""
    return cfg.get("project", {}).get("seed", 42)
