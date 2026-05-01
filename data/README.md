# Data Directory

This directory holds data files for RF-Sentinel.

## Structure

```
data/
├── raw/           # Place the RadioML 2018.01A HDF5 file here
│   └── GOLD_XYZ_OSC.0001_1024.hdf5  (not included — download separately)
├── processed/     # Preprocessed data cache (auto-generated)
└── splits/        # Train/val/test split indices (auto-generated)
    ├── train_indices.npy
    ├── val_indices.npy
    └── test_indices.npy
```

## Dataset Setup

1. Download the **DeepSig RadioML 2018.01A** dataset
2. Place the HDF5 file in `data/raw/`
3. Update `data.dataset_path` in your config file if the filename differs

The dataset is **not** included in this repository due to its size and license terms.
