# Configuration Guide

RF-Sentinel uses YAML configuration files located in `configs/`.

## Config Files

| File | Purpose |
|------|---------|
| `cnn1d_baseline.yaml` | Full CNN1D training on RadioML |
| `cnn1d_quick_test.yaml` | Small-subset smoke test |
| `resnet1d_baseline.yaml` | Full ResNet1D training on RadioML |
| `resnet1d_quick_test.yaml` | Small-subset ResNet1D smoke training |
| `tcn1d_baseline.yaml` | Full TCN1D / dilated CNN training on RadioML |
| `tcn1d_quick_test.yaml` | Small-subset TCN1D smoke training |
| `xgboost_features.yaml` | XGBoost feature-based baseline with a 5000-sample default subset |
| `inference_default.yaml` | Inference-only settings |

## Config Sections

### `project`
| Key | Type | Description |
|-----|------|-------------|
| `name` | str | Project name |
| `seed` | int | Global random seed |
| `output_dir` | str | Artifacts output directory |
| `report_dir` | str | Reports output directory |

Use a model-specific `report_dir` such as `reports/resnet1d` or `reports/tcn1d`
when you want reports from different model families to avoid overwriting each
other.

### `data`
| Key | Type | Description |
|-----|------|-------------|
| `dataset_path` | str | Path to HDF5 dataset file |
| `x_key` | str | HDF5 key for I/Q data |
| `y_key` | str | HDF5 key for labels |
| `snr_key` | str | HDF5 key for SNR values |
| `sample_length` | int | Expected sample length |
| `iq_channels` | int | Number of I/Q channels (2) |
| `max_samples` | int/null | Limit total samples (null = all) |
| `subset_fraction` | float/null | Fraction of data to use |
| `normalize` | str | Normalization: "rms", "zscore", "none" |
| `split_dir` | str | Directory for split indices |
| `reuse_splits` | bool | Reuse existing splits if valid |

If both `max_samples` and `subset_fraction` are `null` and the HDF5 `X` array is
large, RF-Sentinel keeps `X` on disk and uses lazy HDF5 reads during PyTorch waveform training
and evaluation. Subset configs load only the selected samples into memory.

### `splits`
| Key | Type | Description |
|-----|------|-------------|
| `train_size` | float | Training set fraction |
| `val_size` | float | Validation set fraction |
| `test_size` | float | Test set fraction |
| `stratify_by` | str | "label_snr", "label", or "none" |
| `random_seed` | int | Seed for splitting |

### `model`
| Key | Type | Description |
|-----|------|-------------|
| `type` | str | Model type: "cnn1d", "resnet1d", "tcn1d", or "xgboost" |
| `input_channels` | int | Input channels (2) |
| `num_classes` | int/str | Number of classes ("auto" = detect) |
| `dropout` | float | Dropout probability |
| `base_channels` | int | First/primary conv channel width for PyTorch waveform models |
| `blocks_per_stage` | list[int] | ResNet1D only: residual blocks in each of the three stages, e.g. `[2, 2, 2]` |
| `num_blocks` | int | TCN1D only: number of dilated temporal blocks |
| `kernel_size` | int | TCN1D only: odd convolution kernel size used for same-length padding |
| `dilation_base` | int | TCN1D only: exponential dilation base, usually `2` |
| `device` | str | XGBoost only: "auto", "cuda", or "cpu" |
| `gpu_fallback` | bool | XGBoost only: retry on CPU if GPU training is unavailable |
| `tree_method` | str | XGBoost only: tree algorithm, usually "hist" |

Example ResNet1D model section:

```yaml
model:
  type: resnet1d
  input_channels: 2
  num_classes: auto
  base_channels: 64
  dropout: 0.20
  blocks_per_stage: [2, 2, 2]
```

Example TCN1D model section:

```yaml
model:
  type: tcn1d
  input_channels: 2
  num_classes: auto
  base_channels: 64
  dropout: 0.20
  num_blocks: 5
  kernel_size: 5
  dilation_base: 2
```

### `training`
| Key | Type | Description |
|-----|------|-------------|
| `epochs` | int | Maximum training epochs |
| `batch_size` | int | Batch size |
| `learning_rate` | float | Initial learning rate |
| `weight_decay` | float | L2 regularization |
| `optimizer` | str | Optimizer: "adam" |
| `scheduler` | str | LR scheduler: "reduce_on_plateau" |
| `early_stopping_patience` | int | Early stopping patience |
| `num_workers` | int | DataLoader workers |
| `device` | str | "auto", "cpu", or "cuda" |

### `evaluation`
| Key | Type | Description |
|-----|------|-------------|
| `confidence_threshold` | float | PyTorch waveform inference/evaluation: accepted/uncertain threshold |
| `top_k` | int | PyTorch waveform inference: top-k predictions |
| `low_snr_max` | float | Waveform-model robustness analysis: max SNR for "low" band (dB) |
| `high_snr_min` | float | Waveform-model robustness analysis: min SNR for "high" band (dB) |
| `report_subdir` | str | XGBoost only: subdirectory under `project.report_dir` for baseline reports |
| `save_confusion_matrix` | bool | Save confusion matrix figure |
| `save_snr_curve` | bool | Save accuracy-vs-SNR figure |

### `mlflow`
| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | Enable MLflow tracking |
| `tracking_uri` | str | MLflow tracking URI |
| `experiment_name` | str | Experiment name |
| `run_name` | str | Run name |

## Environment Variable Override

```bash
export RF_SENTINEL_DATASET_PATH=/custom/path/to/dataset.hdf5
```

This overrides `data.dataset_path` in any config.
