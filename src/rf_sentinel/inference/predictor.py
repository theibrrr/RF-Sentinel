"""Confidence-aware predictor for trained RF-Sentinel waveform models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional

from rf_sentinel.data.preprocessing import prepare_single_sample
from rf_sentinel.models.factory import build_waveform_model
from rf_sentinel.utils.device import get_device
from rf_sentinel.utils.logging import get_logger, print_info

logger = get_logger(__name__)


class RFPredictor:
    """Confidence-aware predictor using a trained waveform-model checkpoint.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the saved checkpoint .pt file.
    device_cfg : str
        Device configuration ("auto", "cpu", "cuda").
    confidence_threshold : float
        Threshold for accepted/uncertain decision.
    top_k : int
        Number of top predictions to return.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device_cfg: str = "auto",
        confidence_threshold: float = 0.70,
        top_k: int = 3,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = get_device(device_cfg)
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k

        # Load checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(
            str(self.checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )

        self.class_names = checkpoint["class_names"]
        self.index_to_label = {int(k): v for k, v in checkpoint["index_to_label"].items()}
        self.normalization = checkpoint.get("normalization", "rms")
        self.sample_length = checkpoint.get("sample_length", 1024)
        self.num_classes = checkpoint["num_classes"]
        checkpoint_model_type = checkpoint.get(
            "model_type",
            checkpoint.get("model_config", {}).get("type", "cnn1d"),
        )
        self.model_type = str(checkpoint_model_type).lower()

        # Build model
        model_cfg = checkpoint.get("model_config")
        if model_cfg is None:
            model_cfg = {
                "type": self.model_type,
                "input_channels": checkpoint.get("input_channels", 2),
                "num_classes": self.num_classes,
                "base_channels": checkpoint.get("base_channels", 64),
            }
            if checkpoint.get("dropout") is not None:
                model_cfg["dropout"] = checkpoint["dropout"]
        model_cfg = {
            **model_cfg,
            "type": model_cfg.get("type", self.model_type),
            "num_classes": self.num_classes,
        }
        self.model = build_waveform_model({"model": model_cfg})
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        print_info(
            f"Loaded model from {self.checkpoint_path} "
            f"(type={self.model_type}, classes={self.num_classes}, norm={self.normalization})"
        )

    @torch.no_grad()
    def predict(self, sample: np.ndarray) -> dict:
        """Run prediction on a single I/Q sample.

        Parameters
        ----------
        sample : np.ndarray
            Single I/Q sample, shape (L, 2) or (2, L).

        Returns
        -------
        dict
            Prediction result with class, confidence, top-k, and decision.
        """
        # Preprocess
        x = prepare_single_sample(
            sample,
            normalize_method=self.normalization,
            expected_length=self.sample_length,
        )
        x_tensor = torch.from_numpy(x).float().to(self.device)

        # Forward pass
        logits = self.model(x_tensor)
        probs = torch_functional.softmax(logits, dim=1).cpu().numpy()[0]

        # Top-k
        top_n = min(self.top_k, self.num_classes)
        top_indices = np.argsort(probs)[::-1][:top_n]
        top_k_results = []
        for idx in top_indices:
            top_k_results.append(
                {
                    "class": self.index_to_label.get(int(idx), f"Class_{idx}"),
                    "probability": round(float(probs[idx]), 4),
                }
            )

        predicted_idx = int(top_indices[0])
        confidence = float(probs[predicted_idx])
        decision = "accepted" if confidence >= self.confidence_threshold else "uncertain"

        return {
            "prediction": self.index_to_label.get(predicted_idx, f"Class_{predicted_idx}"),
            "confidence": round(confidence, 4),
            "decision": decision,
            "top_k": top_k_results,
        }

    @torch.no_grad()
    def predict_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run batch prediction, returning (predictions, probabilities).

        Parameters
        ----------
        X : np.ndarray
            Preprocessed batch, shape (N, 2, L), float32.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (predicted labels shape (N,), probability matrix shape (N, C))
        """
        if X.ndim != 3 or X.shape[1] != 2:
            raise ValueError(f"Expected preprocessed batch shape (N, 2, L), got {X.shape}.")
        if X.shape[2] != self.sample_length:
            raise ValueError(f"Expected sample length {self.sample_length}, got {X.shape[2]}.")

        self.model.eval()

        all_preds = []
        all_probs = []
        batch_size = 512

        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i : i + batch_size]).float().to(self.device)
            logits = self.model(batch)
            probs = torch_functional.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.append(preds)
            all_probs.append(probs)

        return np.concatenate(all_preds), np.concatenate(all_probs)


def run_single_inference(
    checkpoint_path: str | Path,
    sample_path: str | Path,
    cfg: dict,
) -> dict:
    """Run inference on a single .npy sample file."""
    sample = np.load(str(sample_path))
    eval_cfg = cfg.get("evaluation", {})
    device_cfg = cfg.get("training", {}).get("device", "auto")

    predictor = RFPredictor(
        checkpoint_path=checkpoint_path,
        device_cfg=device_cfg,
        confidence_threshold=eval_cfg.get("confidence_threshold", 0.70),
        top_k=eval_cfg.get("top_k", 3),
    )

    result = predictor.predict(sample)
    print_info(f"Prediction: {json.dumps(result, indent=2)}")
    return result
