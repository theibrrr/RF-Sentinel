"""Loss functions for training."""

from __future__ import annotations

import torch.nn as nn


def get_loss_function(cfg: dict | None = None) -> nn.Module:
    """Get the loss function for training.

    Currently uses CrossEntropyLoss. Can be extended for weighted loss etc.
    """
    return nn.CrossEntropyLoss()
