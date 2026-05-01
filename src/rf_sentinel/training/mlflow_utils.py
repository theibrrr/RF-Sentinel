"""MLflow experiment tracking utilities."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

from rf_sentinel.utils.logging import get_logger, print_info, print_warning

logger = get_logger(__name__)

try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class MLflowTracker:
    """Wrapper for MLflow tracking operations."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        mlflow_cfg = cfg.get("mlflow", {})
        self.enabled = mlflow_cfg.get("enabled", False) and HAS_MLFLOW
        self.tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
        self.experiment_name = mlflow_cfg.get("experiment_name", "rf-sentinel")
        self.run_name = mlflow_cfg.get("run_name", "run")
        self._run = None

        if self.enabled:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            print_info(
                f"MLflow enabled: uri={self.tracking_uri}, experiment={self.experiment_name}"
            )
        elif not HAS_MLFLOW:
            print_warning("MLflow not installed. Tracking disabled.")
        else:
            print_info("MLflow tracking disabled in config.")

    def start_run(self) -> None:
        if self.enabled:
            self._run = mlflow.start_run(run_name=self.run_name)
            print_info(f"MLflow run started: {self.run_name}")

    def end_run(self) -> None:
        if self.enabled and self._run:
            mlflow.end_run()
            print_info("MLflow run ended.")

    def log_params(self, params: dict) -> None:
        if self.enabled:
            for key, value in params.items():
                with suppress(Exception):
                    mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if self.enabled:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if self.enabled:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        if self.enabled:
            path = Path(path)
            if path.exists():
                try:
                    mlflow.log_artifact(str(path))
                except Exception as exc:
                    print_warning(f"Could not log MLflow artifact '{path}': {exc}")

    def log_config_params(self) -> None:
        """Log key configuration parameters."""
        if not self.enabled:
            return

        params = {}
        model_cfg = self.cfg.get("model", {})
        params["model_type"] = model_cfg.get("type", "cnn1d")
        params["base_channels"] = model_cfg.get("base_channels", 64)
        params["dropout"] = model_cfg.get("dropout", 0.25)
        if "blocks_per_stage" in model_cfg:
            params["blocks_per_stage"] = str(model_cfg.get("blocks_per_stage"))
        if "num_blocks" in model_cfg:
            params["num_blocks"] = model_cfg.get("num_blocks")
        if "kernel_size" in model_cfg:
            params["kernel_size"] = model_cfg.get("kernel_size")
        if "dilation_base" in model_cfg:
            params["dilation_base"] = model_cfg.get("dilation_base")

        train_cfg = self.cfg.get("training", {})
        params["learning_rate"] = train_cfg.get("learning_rate", 0.001)
        params["batch_size"] = train_cfg.get("batch_size", 256)
        params["epochs"] = train_cfg.get("epochs", 30)
        params["optimizer"] = train_cfg.get("optimizer", "adam")

        data_cfg = self.cfg.get("data", {})
        params["normalization"] = data_cfg.get("normalize", "rms")
        params["max_samples"] = str(data_cfg.get("max_samples", "all"))

        split_cfg = self.cfg.get("splits", {})
        params["split_seed"] = split_cfg.get("random_seed", 42)

        self.log_params(params)
