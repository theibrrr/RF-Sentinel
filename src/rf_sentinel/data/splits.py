"""Train/validation/test split generation with stratification and persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from rf_sentinel.utils.logging import get_logger, print_info, print_warning
from rf_sentinel.utils.paths import ensure_dir

logger = get_logger(__name__)


def _make_stratification_key(labels, snr, strategy):
    if strategy == "label_snr":
        snr_int = snr.astype(int)
        return np.array(
            [f"{label}_{snr_value}" for label, snr_value in zip(labels, snr_int, strict=True)]
        )
    elif strategy == "label":
        return labels
    return None


def _split_indices(indices, stratify_key, val_size, test_size, seed):
    val_test_ratio = val_size + test_size
    train_idx, valtest_idx = train_test_split(
        indices,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=stratify_key,
    )
    test_frac = test_size / val_test_ratio
    vt_strat = stratify_key[valtest_idx] if stratify_key is not None else None
    val_idx, test_idx = train_test_split(
        valtest_idx,
        test_size=test_frac,
        random_state=seed,
        stratify=vt_strat,
    )
    return train_idx, val_idx, test_idx


def create_splits(n_samples, labels, snr, cfg):
    split_cfg = cfg.get("splits", {})
    val_size = split_cfg.get("val_size", 0.15)
    test_size = split_cfg.get("test_size", 0.15)
    seed = split_cfg.get("random_seed", 42)
    strategy = split_cfg.get("stratify_by", "label_snr")

    indices = np.arange(n_samples)
    stratify_key = _make_stratification_key(labels, snr, strategy)

    try:
        train_idx, val_idx, test_idx = _split_indices(
            indices, stratify_key, val_size, test_size, seed
        )
    except ValueError as e:
        if strategy == "label_snr":
            print_warning(f"Label+SNR stratification failed ({e}). Falling back to label-only.")
            stratify_key = _make_stratification_key(labels, snr, "label")
            try:
                train_idx, val_idx, test_idx = _split_indices(
                    indices, stratify_key, val_size, test_size, seed
                )
            except ValueError as label_error:
                print_warning(
                    f"Label-only stratification failed ({label_error}). "
                    "Falling back to unstratified split."
                )
                train_idx, val_idx, test_idx = _split_indices(
                    indices, None, val_size, test_size, seed
                )
        else:
            raise

    print_info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx


def save_splits(train_idx, val_idx, test_idx, split_dir):
    split_dir = ensure_dir(split_dir)
    np.save(split_dir / "train_indices.npy", train_idx)
    np.save(split_dir / "val_indices.npy", val_idx)
    np.save(split_dir / "test_indices.npy", test_idx)
    print_info(f"Saved split indices to {split_dir}")


def load_splits(split_dir):
    split_dir = Path(split_dir)
    train_idx = np.load(split_dir / "train_indices.npy")
    val_idx = np.load(split_dir / "val_indices.npy")
    test_idx = np.load(split_dir / "test_indices.npy")
    print_info(f"Loaded splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx


def splits_exist(split_dir):
    split_dir = Path(split_dir)
    return all(
        (split_dir / f).exists()
        for f in ["train_indices.npy", "val_indices.npy", "test_indices.npy"]
    )


def get_or_create_splits(n_samples, labels, snr, cfg):
    data_cfg = cfg.get("data", {})
    split_dir = data_cfg.get("split_dir", "data/splits")
    reuse = data_cfg.get("reuse_splits", True)

    if reuse and splits_exist(split_dir):
        train_idx, val_idx, test_idx = load_splits(split_dir)
        total = len(train_idx) + len(val_idx) + len(test_idx)
        if total == n_samples:
            return train_idx, val_idx, test_idx
        print_warning(
            f"Saved splits have {total} indices but dataset has {n_samples}. Regenerating."
        )

    train_idx, val_idx, test_idx = create_splits(n_samples, labels, snr, cfg)
    save_splits(train_idx, val_idx, test_idx, split_dir)
    return train_idx, val_idx, test_idx
