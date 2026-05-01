"""Optional XGBoost baseline using engineered I/Q features."""

from __future__ import annotations

import numpy as np

from rf_sentinel.utils.logging import get_logger, print_error, print_info, print_warning

logger = get_logger(__name__)

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def check_xgboost_available() -> bool:
    if not HAS_XGBOOST:
        print_error(
            "XGBoost is not installed. Install it with:\n"
            "  pip install xgboost\n"
            "Or: pip install -e '.[xgboost]'"
        )
    return HAS_XGBOOST


def _xgboost_major_version() -> int:
    """Return the installed XGBoost major version, best-effort."""
    try:
        return int(str(xgb.__version__).split(".", maxsplit=1)[0])
    except Exception:
        return 2


def _torch_cuda_available() -> bool:
    """Check CUDA availability without making XGBoost depend directly on PyTorch imports."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_xgboost_device(device_cfg: str) -> str:
    """Resolve XGBoost device config to 'cuda' or 'cpu'."""
    if device_cfg == "auto":
        if _torch_cuda_available():
            return "cuda"
        print_info("CUDA not available to PyTorch; XGBoost will use CPU.")
        return "cpu"
    if device_cfg in {"cuda", "gpu"}:
        if not _torch_cuda_available():
            print_warning("XGBoost GPU was requested, but CUDA is not available. Using CPU.")
            return "cpu"
        return "cuda"
    return "cpu"


def build_xgboost(cfg: dict) -> xgb.XGBClassifier | None:
    """Build an XGBoost classifier from config."""
    if not check_xgboost_available():
        return None

    model_cfg = cfg.get("model", {})
    num_classes = model_cfg.get("num_classes", 24)
    if num_classes == "auto":
        num_classes = 24

    resolved_device = _resolve_xgboost_device(model_cfg.get("device", "auto"))
    tree_method = model_cfg.get("tree_method", "hist")

    params = {
        "n_estimators": model_cfg.get("n_estimators", 300),
        "max_depth": model_cfg.get("max_depth", 8),
        "learning_rate": model_cfg.get("learning_rate", 0.1),
        "subsample": model_cfg.get("subsample", 0.8),
        "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "random_state": cfg.get("project", {}).get("seed", 42),
        "verbosity": model_cfg.get("verbosity", 1),
    }

    if _xgboost_major_version() >= 2:
        params["tree_method"] = tree_method
        params["device"] = resolved_device
    else:
        params["tree_method"] = "gpu_hist" if resolved_device == "cuda" else tree_method
        if resolved_device == "cuda":
            params["predictor"] = "gpu_predictor"

    clf = xgb.XGBClassifier(**params)
    print_info(
        "Built XGBoost classifier: "
        f"n_estimators={clf.n_estimators}, max_depth={clf.max_depth}, "
        f"device={resolved_device}, tree_method={params.get('tree_method')}"
    )
    return clf


def _uses_xgboost_gpu(clf: xgb.XGBClassifier) -> bool:
    params = clf.get_params()
    return params.get("device") == "cuda" or params.get("tree_method") == "gpu_hist"


def _force_xgboost_cpu(clf: xgb.XGBClassifier) -> xgb.XGBClassifier:
    params = clf.get_params()
    params["tree_method"] = "hist"
    params.pop("predictor", None)
    if "device" in params:
        params["device"] = "cpu"
    return xgb.XGBClassifier(**params)


def train_xgboost(
    clf: xgb.XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    fallback_to_cpu: bool = True,
) -> xgb.XGBClassifier:
    """Train the XGBoost classifier."""
    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))

    try:
        clf.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
    except Exception as exc:
        if not fallback_to_cpu or not _uses_xgboost_gpu(clf):
            raise
        print_warning(f"XGBoost GPU training failed ({exc}). Retrying on CPU.")
        clf = _force_xgboost_cpu(clf)
        clf.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
    print_info("XGBoost training complete.")
    return clf
