"""RF-Sentinel CLI — Typer-based command-line interface.

Commands:
    inspect-data       Inspect the RadioML dataset
    make-splits        Generate train/val/test splits
    train              Train the configured waveform model
    evaluate           Evaluate trained model
    infer              Inference on a single .npy sample
    infer-from-dataset Inference on a dataset sample by index
    train-xgboost      Train XGBoost baseline
    smoke-test         Run smoke test with synthetic data
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

console = Console()
app = typer.Typer(
    name="rf-sentinel",
    help="RF-Sentinel: Robust RF Modulation Recognition from Raw I/Q Data",
    add_completion=False,
)


@app.command()
def inspect_data(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
):
    """Inspect the RadioML dataset and generate summary reports."""
    from rf_sentinel.config.loader import load_config
    from rf_sentinel.data.inspection import run_inspection
    from rf_sentinel.utils.logging import print_header

    print_header("Dataset Inspection", "RF-Sentinel")
    cfg = load_config(config)
    run_inspection(cfg)
    console.print("[bold green]Dataset inspection complete.[/bold green]")


@app.command()
def make_splits(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
):
    """Generate train/validation/test split indices."""
    from rf_sentinel.config.loader import load_config
    from rf_sentinel.data.radioml_loader import load_radioml
    from rf_sentinel.data.splits import create_splits, save_splits
    from rf_sentinel.utils.logging import print_header
    from rf_sentinel.utils.reproducibility import get_seed_from_config, set_global_seed

    print_header("Generate Splits", "RF-Sentinel")
    cfg = load_config(config)
    set_global_seed(get_seed_from_config(cfg))
    dataset = load_radioml(cfg)

    train_idx, val_idx, test_idx = create_splits(
        len(dataset),
        dataset.labels,
        dataset.snr,
        cfg,
    )
    split_dir = cfg.get("data", {}).get("split_dir", "data/splits")
    save_splits(train_idx, val_idx, test_idx, split_dir)
    console.print("[bold green]Splits generated and saved.[/bold green]")


@app.command()
def train(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
):
    """Train a raw-I/Q waveform modulation classifier."""
    from rf_sentinel.config.loader import load_config
    from rf_sentinel.data.radioml_loader import load_radioml
    from rf_sentinel.training.trainer import run_training
    from rf_sentinel.utils.logging import print_header

    print_header("Model Training", "RF-Sentinel")
    cfg = load_config(config)
    dataset = load_radioml(cfg)
    best_path = run_training(dataset, cfg)
    console.print(f"[bold green]Training complete. Best model: {best_path}[/bold green]")


@app.command()
def evaluate(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-k", help="Path to model checkpoint"),
):
    """Evaluate a trained waveform model on the test set."""
    from torch.utils.data import DataLoader

    from rf_sentinel.config.loader import load_config
    from rf_sentinel.data.dataset import HDF5IQDataset
    from rf_sentinel.data.preprocessing import preprocess_dataset
    from rf_sentinel.data.radioml_loader import load_radioml
    from rf_sentinel.data.splits import get_or_create_splits
    from rf_sentinel.evaluation.error_analysis import generate_error_analysis_report
    from rf_sentinel.evaluation.metrics import (
        compute_classification_metrics,
        compute_confidence_metrics,
        save_classification_report_csv,
        save_evaluation_metrics,
    )
    from rf_sentinel.evaluation.plots import (
        plot_accuracy_vs_snr,
        plot_confidence_histogram,
        plot_confusion_matrix,
    )
    from rf_sentinel.evaluation.robustness import (
        compute_per_snr_accuracy,
        compute_snr_band_metrics,
        save_per_snr_accuracy_csv,
    )
    from rf_sentinel.inference.predictor import RFPredictor
    from rf_sentinel.training.mlflow_utils import MLflowTracker
    from rf_sentinel.utils.logging import print_header, print_info, print_success
    from rf_sentinel.utils.paths import ensure_dir
    from rf_sentinel.utils.reproducibility import get_seed_from_config, set_global_seed

    print_header("Model Evaluation", "RF-Sentinel")
    cfg = load_config(config)
    set_global_seed(get_seed_from_config(cfg))

    eval_cfg = cfg.get("evaluation", {})
    report_dir = cfg.get("project", {}).get("report_dir", "reports")
    figures_dir = ensure_dir(Path(report_dir) / "figures")

    # Load dataset and splits
    dataset = load_radioml(cfg)
    _, _, test_idx = get_or_create_splits(len(dataset), dataset.labels, dataset.snr, cfg)

    y_true = dataset.labels[test_idx]
    snr_test = dataset.snr[test_idx]

    # Load predictor and run batch prediction
    predictor = RFPredictor(
        checkpoint_path=checkpoint,
        device_cfg=cfg.get("training", {}).get("device", "auto"),
        confidence_threshold=eval_cfg.get("confidence_threshold", 0.70),
        top_k=eval_cfg.get("top_k", 3),
    )

    normalize_method = cfg.get("data", {}).get("normalize", "rms")
    if dataset.X is None:
        if dataset.dataset_path is None:
            raise ValueError("Lazy HDF5 evaluation requires dataset_path metadata.")
        test_ds = HDF5IQDataset(
            dataset.dataset_path,
            dataset.x_key,
            test_idx,
            y_true,
            snr_test,
            normalize_method=normalize_method,
            sample_length=dataset.sample_length,
        )
        batch_size = cfg.get("training", {}).get("batch_size", 256)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        pred_batches = []
        prob_batches = []
        for batch in test_loader:
            preds, probs = predictor.predict_batch(batch["x"].numpy())
            pred_batches.append(preds)
            prob_batches.append(probs)
        y_pred = np.concatenate(pred_batches)
        probabilities = np.concatenate(prob_batches)
    else:
        X_processed = preprocess_dataset(dataset.X[test_idx], normalize_method)
        y_pred, probabilities = predictor.predict_batch(X_processed)

    # Compute metrics
    class_metrics = compute_classification_metrics(y_true, y_pred, dataset.class_names)
    conf_metrics = compute_confidence_metrics(
        probabilities,
        y_true,
        y_pred,
        eval_cfg.get("confidence_threshold", 0.70),
    )
    per_snr = compute_per_snr_accuracy(y_true, y_pred, snr_test)
    snr_bands = compute_snr_band_metrics(
        y_true,
        y_pred,
        snr_test,
        eval_cfg.get("low_snr_max", 0),
        eval_cfg.get("high_snr_min", 12),
    )

    # Combine all metrics
    all_metrics = {**class_metrics, **conf_metrics, **snr_bands}
    all_metrics.pop("confusion_matrix", None)  # too large for JSON readability
    all_metrics.pop("per_class_report", None)

    print_info(f"Overall Accuracy: {class_metrics['accuracy']:.4f}")
    print_info(f"Macro F1: {class_metrics['macro_f1']:.4f}")
    print_info(f"Weighted F1: {class_metrics['weighted_f1']:.4f}")
    print_info(f"Accepted ratio: {conf_metrics['accepted_ratio']:.4f}")
    if snr_bands.get("low_snr_accuracy") is not None:
        print_info(f"Low-SNR accuracy: {snr_bands['low_snr_accuracy']:.4f}")
    if snr_bands.get("high_snr_accuracy") is not None:
        print_info(f"High-SNR accuracy: {snr_bands['high_snr_accuracy']:.4f}")

    # Save reports
    save_evaluation_metrics(all_metrics, report_dir)
    save_classification_report_csv(y_true, y_pred, dataset.class_names, report_dir)
    save_per_snr_accuracy_csv(per_snr, report_dir)

    # Save plots
    cm = np.array(class_metrics["confusion_matrix"])
    plot_confusion_matrix(cm, dataset.class_names, figures_dir / "confusion_matrix.png")
    plot_accuracy_vs_snr(
        list(per_snr.keys()),
        list(per_snr.values()),
        figures_dir / "accuracy_vs_snr.png",
    )
    confidences = np.max(probabilities, axis=1)
    plot_confidence_histogram(
        confidences,
        eval_cfg.get("confidence_threshold", 0.70),
        figures_dir / "confidence_histogram.png",
    )

    # Error analysis
    generate_error_analysis_report(
        y_true,
        y_pred,
        snr_test,
        dataset.class_names,
        cfg,
        report_dir,
    )

    # MLflow logging
    tracker = MLflowTracker(cfg)
    tracker.start_run()
    tracker.log_metrics(
        {
            "test_accuracy": class_metrics["accuracy"],
            "macro_f1": class_metrics["macro_f1"],
            "weighted_f1": class_metrics["weighted_f1"],
            "accepted_ratio": conf_metrics["accepted_ratio"],
            "uncertain_ratio": conf_metrics["uncertain_ratio"],
        }
    )
    if snr_bands.get("low_snr_accuracy") is not None:
        tracker.log_metric("low_snr_accuracy", snr_bands["low_snr_accuracy"])
    if snr_bands.get("high_snr_accuracy") is not None:
        tracker.log_metric("high_snr_accuracy", snr_bands["high_snr_accuracy"])

    tracker.log_artifact(Path(report_dir) / "evaluation_metrics.json")
    tracker.log_artifact(Path(report_dir) / "classification_report.csv")
    tracker.log_artifact(Path(report_dir) / "error_analysis.md")
    tracker.log_artifact(figures_dir / "confusion_matrix.png")
    tracker.log_artifact(figures_dir / "accuracy_vs_snr.png")
    tracker.log_artifact(figures_dir / "confidence_histogram.png")
    tracker.end_run()

    print_success("Evaluation complete.")


@app.command()
def infer(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-k", help="Path to model checkpoint"),
    sample: str = typer.Option(..., "--sample", "-s", help="Path to .npy sample file"),
):
    """Run inference on a single .npy I/Q sample."""
    from rf_sentinel.config.loader import load_config
    from rf_sentinel.inference.predictor import run_single_inference
    from rf_sentinel.utils.logging import print_header

    print_header("Single Sample Inference", "RF-Sentinel")
    cfg = load_config(config)
    result = run_single_inference(checkpoint, sample, cfg)
    console.print_json(json.dumps(result))


@app.command()
def infer_from_dataset(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-k", help="Path to model checkpoint"),
    index: int = typer.Option(..., "--index", "-i", help="Sample index in dataset"),
):
    """Run inference on a sample from the dataset by index."""
    import h5py

    from rf_sentinel.config.loader import load_config
    from rf_sentinel.inference.predictor import RFPredictor
    from rf_sentinel.utils.logging import print_header, print_info

    print_header("Dataset Sample Inference", "RF-Sentinel")
    cfg = load_config(config)
    data_cfg = cfg["data"]

    dataset_path = data_cfg["dataset_path"]
    if not Path(dataset_path).exists():
        console.print(f"[bold red]Dataset not found: {dataset_path}[/bold red]")
        raise typer.Exit(1)

    x_key = data_cfg.get("x_key", "X")
    y_key = data_cfg.get("y_key", "Y")
    snr_key = data_cfg.get("snr_key", "Z")
    with h5py.File(dataset_path, "r") as f:
        missing = [key for key in [x_key, y_key, snr_key] if key not in f]
        if missing:
            console.print(f"[bold red]Missing HDF5 key(s): {missing}[/bold red]")
            raise typer.Exit(1)
        n_samples = f[x_key].shape[0]
        if index < 0 or index >= n_samples:
            console.print(
                f"[bold red]Index {index} out of range for dataset with {n_samples} samples.[/bold red]"
            )
            raise typer.Exit(1)

        sample = f[x_key][index].astype(np.float32)
        y_raw = f[y_key][index]
        snr_val = float(f[snr_key][index].flatten()[0])

    true_label_idx = int(np.argmax(y_raw)) if y_raw.ndim >= 1 and len(y_raw) > 1 else int(y_raw)

    eval_cfg = cfg.get("evaluation", {})
    predictor = RFPredictor(
        checkpoint_path=checkpoint,
        device_cfg=cfg.get("training", {}).get("device", "auto"),
        confidence_threshold=eval_cfg.get("confidence_threshold", 0.70),
        top_k=eval_cfg.get("top_k", 3),
    )

    true_label = predictor.index_to_label.get(true_label_idx, f"Class_{true_label_idx}")
    print_info(f"Sample index: {index}")
    print_info(f"True label: {true_label} (idx={true_label_idx})")
    print_info(f"SNR: {snr_val} dB")

    result = predictor.predict(sample)
    result["true_label"] = true_label
    result["snr_db"] = snr_val
    result["correct"] = result["prediction"] == true_label

    console.print_json(json.dumps(result))


@app.command()
def train_xgboost(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
):
    """Train XGBoost baseline using engineered I/Q features."""
    from rf_sentinel.config.loader import load_config
    from rf_sentinel.data.radioml_loader import load_radioml
    from rf_sentinel.data.splits import get_or_create_splits
    from rf_sentinel.evaluation.metrics import (
        compute_classification_metrics,
        save_classification_report_csv,
        save_evaluation_metrics,
    )
    from rf_sentinel.evaluation.plots import plot_accuracy_vs_snr, plot_confusion_matrix
    from rf_sentinel.evaluation.robustness import (
        compute_per_snr_accuracy,
        save_per_snr_accuracy_csv,
    )
    from rf_sentinel.models.xgboost_baseline import build_xgboost, check_xgboost_available
    from rf_sentinel.models.xgboost_baseline import train_xgboost as _train_xgb
    from rf_sentinel.signal.iq_features import extract_features_batch
    from rf_sentinel.training.mlflow_utils import MLflowTracker
    from rf_sentinel.utils.logging import print_header, print_info, print_success
    from rf_sentinel.utils.paths import ensure_dir
    from rf_sentinel.utils.reproducibility import get_seed_from_config, set_global_seed

    print_header("XGBoost Training", "RF-Sentinel")

    if not check_xgboost_available():
        raise typer.Exit(1)

    cfg = load_config(config)
    set_global_seed(get_seed_from_config(cfg))

    dataset = load_radioml(cfg)
    if dataset.X is None:
        console.print(
            "[bold red]XGBoost baseline requires in-memory samples. Set data.max_samples "
            "or data.subset_fraction in the XGBoost config for this large dataset.[/bold red]"
        )
        raise typer.Exit(1)

    train_idx, val_idx, test_idx = get_or_create_splits(
        len(dataset),
        dataset.labels,
        dataset.snr,
        cfg,
    )

    feature_cfg = cfg.get("features", {})
    feature_set = feature_cfg.get("feature_set", None)
    print_info("Extracting features...")

    X_train_feat = extract_features_batch(dataset.X[train_idx], feature_set)
    X_val_feat = extract_features_batch(dataset.X[val_idx], feature_set)
    print_info(f"Feature shape: {X_train_feat.shape}")

    model_cfg = cfg.get("model", {}).copy()
    model_cfg["num_classes"] = dataset.num_classes
    cfg = {**cfg, "model": model_cfg}
    clf = build_xgboost(cfg)
    clf = _train_xgb(
        clf,
        X_train_feat,
        dataset.labels[train_idx],
        X_val_feat,
        dataset.labels[val_idx],
        fallback_to_cpu=cfg.get("model", {}).get("gpu_fallback", True),
    )

    # Medium evaluation for the baseline: core classification metrics and SNR curves.
    eval_cfg = cfg.get("evaluation", {})
    base_report_dir = Path(cfg.get("project", {}).get("report_dir", "reports"))
    report_subdir = eval_cfg.get("report_subdir", "xgboost")
    report_dir = ensure_dir(base_report_dir / report_subdir)
    figures_dir = ensure_dir(report_dir / "figures")

    X_test_feat = extract_features_batch(dataset.X[test_idx], feature_set)
    y_true = dataset.labels[test_idx]
    snr_test = dataset.snr[test_idx]
    y_pred = clf.predict(X_test_feat)

    class_metrics = compute_classification_metrics(y_true, y_pred, dataset.class_names)
    per_snr = compute_per_snr_accuracy(y_true, y_pred, snr_test)
    eval_summary = {
        "model_type": "xgboost",
        "overall_accuracy": class_metrics["accuracy"],
        "macro_f1": class_metrics["macro_f1"],
        "weighted_f1": class_metrics["weighted_f1"],
        "feature_count": int(X_train_feat.shape[1]),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
    }

    print_info(f"Test accuracy: {class_metrics['accuracy']:.4f}")
    print_info(f"Test macro F1: {class_metrics['macro_f1']:.4f}")
    print_info(f"Test weighted F1: {class_metrics['weighted_f1']:.4f}")
    print_info(f"XGBoost reports: {report_dir}")

    save_evaluation_metrics(eval_summary, report_dir)
    save_classification_report_csv(y_true, y_pred, dataset.class_names, report_dir)
    save_per_snr_accuracy_csv(per_snr, report_dir)

    if eval_cfg.get("save_confusion_matrix", True):
        cm = np.array(class_metrics["confusion_matrix"])
        plot_confusion_matrix(cm, dataset.class_names, figures_dir / "confusion_matrix.png")
    if eval_cfg.get("save_snr_curve", True):
        plot_accuracy_vs_snr(
            list(per_snr.keys()),
            list(per_snr.values()),
            figures_dir / "accuracy_vs_snr.png",
        )

    # MLflow
    tracker = MLflowTracker(cfg)
    tracker.start_run()
    tracker.log_config_params()
    tracker.log_params(
        {
            "feature_count": int(X_train_feat.shape[1]),
            "feature_set": ",".join(feature_set) if feature_set else "all",
            "report_dir": str(report_dir),
            "test_samples": int(len(test_idx)),
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
        }
    )
    tracker.log_metrics(
        {
            "test_accuracy": class_metrics["accuracy"],
            "macro_f1": class_metrics["macro_f1"],
            "weighted_f1": class_metrics["weighted_f1"],
        }
    )
    tracker.log_metrics(
        {
            f"snr_accuracy/snr_{int(snr_db) if float(snr_db).is_integer() else snr_db}_db": acc
            for snr_db, acc in per_snr.items()
        }
    )
    tracker.log_artifact(report_dir / "evaluation_metrics.json")
    tracker.log_artifact(report_dir / "classification_report.csv")
    tracker.log_artifact(report_dir / "per_snr_accuracy.csv")
    tracker.log_artifact(figures_dir / "confusion_matrix.png")
    tracker.log_artifact(figures_dir / "accuracy_vs_snr.png")
    tracker.end_run()

    print_success("XGBoost training complete.")


@app.command()
def smoke_test():
    """Run a quick smoke test with synthetic data (no real dataset needed)."""
    import torch

    from rf_sentinel.data.preprocessing import prepare_single_sample, preprocess_dataset
    from rf_sentinel.models.factory import build_waveform_model, count_parameters
    from rf_sentinel.utils.logging import print_header, print_info, print_success
    from rf_sentinel.utils.reproducibility import set_global_seed

    print_header("Smoke Test", "RF-Sentinel")
    set_global_seed(42)

    # Generate synthetic data
    n_samples = 200
    sample_length = 1024
    num_classes = 8

    print_info(f"Generating {n_samples} synthetic I/Q samples...")
    X_raw = np.random.randn(n_samples, sample_length, 2).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=n_samples)

    # Test preprocessing
    print_info("Testing preprocessing...")
    X_processed = preprocess_dataset(X_raw, normalize_method="rms")
    assert X_processed.shape == (n_samples, 2, sample_length), f"Bad shape: {X_processed.shape}"
    print_success(f"Preprocessing OK: {X_processed.shape}")

    # Test single sample preprocessing
    single = prepare_single_sample(X_raw[0])
    assert single.shape == (1, 2, sample_length), f"Bad single shape: {single.shape}"
    print_success(f"Single sample prep OK: {single.shape}")

    # Test supported waveform models
    print_info("Testing waveform models...")
    model_configs = [
        {"type": "cnn1d", "input_channels": 2, "num_classes": num_classes, "base_channels": 32},
        {
            "type": "resnet1d",
            "input_channels": 2,
            "num_classes": num_classes,
            "base_channels": 16,
            "blocks_per_stage": [1, 1, 1],
        },
        {
            "type": "tcn1d",
            "input_channels": 2,
            "num_classes": num_classes,
            "base_channels": 16,
            "num_blocks": 3,
            "kernel_size": 5,
        },
    ]
    models = {}
    x_batch = torch.from_numpy(X_processed[:16]).float()
    for model_cfg in model_configs:
        model_type = model_cfg["type"]
        model = build_waveform_model({"model": model_cfg})
        n_params = count_parameters(model)
        with torch.no_grad():
            output = model(x_batch)
        assert output.shape == (16, num_classes), f"Bad {model_type} output shape: {output.shape}"
        print_success(f"{model_type} forward OK: {output.shape}, params={n_params:,}")
        models[model_type] = model

    # Test mini training loop
    print_info("Running 1-epoch mini training loop on CNN1D...")
    model = models["cnn1d"]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    x_all = torch.from_numpy(X_processed).float()
    y_all = torch.from_numpy(labels).long()

    for i in range(0, len(x_all), 32):
        batch_x = x_all[i : i + 32]
        batch_y = y_all[i : i + 32]
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

    print_success(f"Mini training loop OK (final loss: {loss.item():.4f})")

    # Test prediction
    print_info("Testing prediction...")
    model.eval()
    with torch.no_grad():
        logits = model(x_all[:1])
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        conf = probs.max().item()

    print_success(f"Prediction OK: class={pred}, confidence={conf:.4f}")

    console.print()
    console.print("[bold green]=== All smoke tests passed! ===[/bold green]")
    console.print()


def main():
    app()


if __name__ == "__main__":
    main()
