"""Training loop for raw I/Q waveform modulation classifiers."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from rf_sentinel.data.dataset import HDF5IQDataset, IQDataset
from rf_sentinel.data.preprocessing import preprocess_dataset
from rf_sentinel.data.radioml_loader import RadioMLDataset
from rf_sentinel.data.splits import get_or_create_splits
from rf_sentinel.models.factory import (
    build_waveform_model,
    checkpoint_stem,
    get_model_type,
    print_model_summary,
)
from rf_sentinel.training.losses import get_loss_function
from rf_sentinel.training.mlflow_utils import MLflowTracker
from rf_sentinel.utils.device import get_device
from rf_sentinel.utils.logging import (
    get_logger,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from rf_sentinel.utils.paths import ensure_dir, get_checkpoint_dir
from rf_sentinel.utils.reproducibility import get_seed_from_config, set_global_seed

logger = get_logger(__name__)


class EarlyStopping:
    """Early stopping monitor based on validation loss."""

    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch, returning average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run validation, returning (loss, accuracy, macro_f1)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        x = batch["x"].to(device)
        labels = batch["label"].to(device)

        logits = model(x)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = float(np.mean(all_preds == all_labels))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

    return avg_loss, accuracy, macro_f1


def _save_checkpoint(
    model: nn.Module,
    cfg: dict,
    class_names: list[str],
    path: Path,
    extra: dict | None = None,
) -> None:
    """Save model checkpoint with metadata."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "class_names": class_names,
        "label_to_index": {name: i for i, name in enumerate(class_names)},
        "index_to_label": {i: name for i, name in enumerate(class_names)},
        "normalization": cfg.get("data", {}).get("normalize", "rms"),
        "sample_length": cfg.get("data", {}).get("sample_length", 1024),
        "model_type": cfg.get("model", {}).get("type", "cnn1d"),
        "model_config": cfg.get("model", {}).copy(),
        "input_channels": getattr(model, "input_channels", 2),
        "num_classes": getattr(model, "num_classes", len(class_names)),
        "base_channels": getattr(model, "base_channels", None),
        "dropout": getattr(model, "dropout", None),
    }
    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, str(path))
    print_info(f"Checkpoint saved: {path}")


def _save_training_curves(history: dict, report_dir: str | Path) -> None:
    """Save training curves as JSON and plot."""
    report_dir = ensure_dir(Path(report_dir))
    figures_dir = ensure_dir(report_dir / "figures")

    # Save history JSON
    with open(report_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Plot curves
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss
        axes[0].plot(history["train_loss"], label="Train")
        axes[0].plot(history["val_loss"], label="Val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(history["val_accuracy"], label="Val Accuracy", color="green")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Macro F1
        axes[2].plot(history["val_macro_f1"], label="Val Macro F1", color="orange")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Macro F1")
        axes[2].set_title("Validation Macro F1")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = figures_dir / "training_curves.png"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print_info(f"Training curves saved: {fig_path}")
    except Exception as e:
        logger.warning(f"Could not save training curves plot: {e}")


def run_training(dataset: RadioMLDataset, cfg: dict) -> Path:
    """Execute the full waveform-model training pipeline.

    Parameters
    ----------
    dataset : RadioMLDataset
        Loaded RadioML dataset.
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    Path
        Path to the best checkpoint.
    """
    model_type = get_model_type(cfg)
    print_header(f"{model_type.upper()} Training Pipeline", "RF-Sentinel")

    seed = get_seed_from_config(cfg)
    set_global_seed(seed)

    train_cfg = cfg.get("training", {})
    device = get_device(train_cfg.get("device", "auto"))

    # Get or create splits
    train_idx, val_idx, _test_idx = get_or_create_splits(
        len(dataset), dataset.labels, dataset.snr, cfg
    )

    # Preprocess or configure lazy HDF5 reads for large full-dataset runs.
    normalize_method = cfg.get("data", {}).get("normalize", "rms")
    print_info(f"Normalization: {normalize_method}")
    num_workers = train_cfg.get("num_workers", 2)
    if dataset.X is None:
        if dataset.dataset_path is None:
            raise ValueError("Lazy HDF5 training requires dataset_path metadata.")
        if os.name == "nt" and num_workers > 0:
            print_warning(
                "Using num_workers=0 for lazy HDF5 training on Windows to avoid "
                "multiprocessing file-handle issues."
            )
            num_workers = 0

        train_ds = HDF5IQDataset(
            dataset.dataset_path,
            dataset.x_key,
            train_idx,
            dataset.labels[train_idx],
            dataset.snr[train_idx],
            normalize_method=normalize_method,
            sample_length=dataset.sample_length,
        )
        val_ds = HDF5IQDataset(
            dataset.dataset_path,
            dataset.x_key,
            val_idx,
            dataset.labels[val_idx],
            dataset.snr[val_idx],
            normalize_method=normalize_method,
            sample_length=dataset.sample_length,
        )
        print_info("Using lazy HDF5 datasets for training and validation.")
    else:
        print_info("Preprocessing in-memory dataset.")
        X_processed = preprocess_dataset(dataset.X, normalize_method)
        train_ds = IQDataset(
            X_processed[train_idx], dataset.labels[train_idx], dataset.snr[train_idx]
        )
        val_ds = IQDataset(X_processed[val_idx], dataset.labels[val_idx], dataset.snr[val_idx])

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 256),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 256),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    # Build model
    model_cfg = cfg.get("model", {}).copy()
    model_cfg["type"] = model_type
    model_cfg["num_classes"] = dataset.num_classes
    data_cfg = cfg.get("data", {}).copy()
    data_cfg["sample_length"] = dataset.sample_length
    cfg_with_classes = {**cfg, "data": data_cfg, "model": model_cfg}
    model = build_waveform_model(cfg_with_classes)
    print_model_summary(model, model_type)
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = get_loss_function(cfg)
    lr = train_cfg.get("learning_rate", 0.001)
    wd = train_cfg.get("weight_decay", 1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = None
    if train_cfg.get("scheduler") == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

    early_stopping = EarlyStopping(patience=train_cfg.get("early_stopping_patience", 7))

    # MLflow tracking
    tracker = MLflowTracker(cfg)
    tracker.start_run()
    tracker.log_config_params()

    # Training loop
    epochs = train_cfg.get("epochs", 30)
    checkpoint_dir = get_checkpoint_dir(cfg)
    best_val_loss = float("inf")
    ckpt_stem = checkpoint_stem(model_type)
    best_checkpoint_path = checkpoint_dir / f"{ckpt_stem}_best.pt"
    report_dir = cfg.get("project", {}).get("report_dir", "reports")

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}

    print_info(f"Training for {epochs} epochs on {device}")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = _validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        # Log to MLflow
        tracker.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            },
            step=epoch,
        )

        print_info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(
                model,
                cfg_with_classes,
                dataset.class_names,
                best_checkpoint_path,
                extra={"epoch": epoch, "val_loss": val_loss, "val_accuracy": val_acc},
            )

        # Early stopping
        if early_stopping(val_loss):
            print_warning(f"Early stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - start_time
    print_success(f"Training complete in {total_time:.1f}s")

    # Save final checkpoint
    final_path = checkpoint_dir / f"{ckpt_stem}_final.pt"
    _save_checkpoint(model, cfg_with_classes, dataset.class_names, final_path)

    # Save training curves
    _save_training_curves(history, report_dir)

    # Save config copy
    import yaml

    config_copy_path = Path(report_dir) / "training_config.yaml"
    with open(config_copy_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Log artifacts to MLflow
    tracker.log_artifact(best_checkpoint_path)
    tracker.log_artifact(config_copy_path)
    tracker.end_run()

    print_success(f"Best checkpoint: {best_checkpoint_path}")
    return best_checkpoint_path
