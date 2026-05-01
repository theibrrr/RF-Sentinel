"""PyTorch Dataset wrapper for preprocessed I/Q data."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from rf_sentinel.data.preprocessing import prepare_single_sample


class IQDataset(Dataset):
    """PyTorch Dataset for preprocessed I/Q samples.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed I/Q data, shape (N, 2, sample_length), float32.
    labels : np.ndarray
        Integer class labels, shape (N,).
    snr : np.ndarray, optional
        SNR values, shape (N,). Stored for evaluation but not used in training.
    """

    def __init__(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        snr: np.ndarray | None = None,
    ):
        self.X = torch.from_numpy(X).float()
        self.labels = torch.from_numpy(labels).long()
        self.snr = torch.from_numpy(snr).float() if snr is not None else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "x": self.X[idx],
            "label": self.labels[idx],
        }
        if self.snr is not None:
            item["snr"] = self.snr[idx]
        return item


class HDF5IQDataset(Dataset):
    """Lazy PyTorch Dataset that reads RadioML I/Q samples from HDF5 on demand.

    This avoids materializing the 20+ GB RadioML X array and a second
    preprocessed copy in memory during full-dataset training/evaluation.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        x_key: str,
        indices: np.ndarray,
        labels: np.ndarray,
        snr: np.ndarray | None = None,
        normalize_method: str = "rms",
        sample_length: int = 1024,
    ):
        self.dataset_path = str(dataset_path)
        self.x_key = x_key
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.snr = np.asarray(snr, dtype=np.float32) if snr is not None else None
        self.normalize_method = normalize_method
        self.sample_length = sample_length
        self._file: h5py.File | None = None
        self._x_dataset: h5py.Dataset | None = None

        if len(self.indices) != len(self.labels):
            raise ValueError(
                f"indices/labels length mismatch: {len(self.indices)} vs {len(self.labels)}"
            )
        if self.snr is not None and len(self.indices) != len(self.snr):
            raise ValueError(f"indices/snr length mismatch: {len(self.indices)} vs {len(self.snr)}")

    def _get_x_dataset(self) -> h5py.Dataset:
        if self._file is None:
            self._file = h5py.File(self.dataset_path, "r")
            self._x_dataset = self._file[self.x_key]
        assert self._x_dataset is not None
        return self._x_dataset

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x_dataset = self._get_x_dataset()
        raw_index = int(self.indices[idx])
        sample = x_dataset[raw_index].astype(np.float32)
        prepared = prepare_single_sample(
            sample,
            normalize_method=self.normalize_method,
            expected_length=self.sample_length,
        )[0]

        item = {
            "x": torch.from_numpy(prepared).float(),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }
        if self.snr is not None:
            item["snr"] = torch.tensor(float(self.snr[idx]), dtype=torch.float32)
        return item

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_x_dataset"] = None
        return state

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()
