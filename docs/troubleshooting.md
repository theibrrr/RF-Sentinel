# Troubleshooting

Common issues and solutions for RF-Sentinel.

## Dataset File Not Found

**Error**: `FileNotFoundError: Dataset file not found: data/raw/GOLD_XYZ_OSC.0001_1024.hdf5`

**Solution**:
1. Download the DeepSig RadioML 2018.01A dataset
2. Place the HDF5 file in `data/raw/`
3. Or update `data.dataset_path` in your config YAML
4. Or set `RF_SENTINEL_DATASET_PATH` environment variable

## Wrong HDF5 Keys

**Error**: `KeyError: Expected HDF5 key 'X' for I/Q data not found`

**Solution**:
1. Run `rf-sentinel inspect-data` to see available keys
2. Update `data.x_key`, `data.y_key`, `data.snr_key` in your config

## Memory Error / Out of Memory

**Error**: `MemoryError` when loading dataset

**Solution**:
1. For PyTorch waveform training, keep `data.max_samples: null` to use lazy HDF5 reads on the full dataset, or use a subset for faster development.
2. Use `data.max_samples` to limit samples:
   ```yaml
   data:
     max_samples: 50000
   ```
3. Use `data.subset_fraction`:
   ```yaml
   data:
     subset_fraction: 0.1
   ```
4. Reduce `training.batch_size`
5. Use a quick test config such as `configs/cnn1d_quick_test.yaml`, `configs/resnet1d_quick_test.yaml`, or `configs/tcn1d_quick_test.yaml`
6. For XGBoost, use a subset; the feature baseline intentionally loads selected samples into memory.

## CUDA Not Available

**Message**: `CUDA not available, using CPU`

**Solution**:
1. Install PyTorch with CUDA support:
   ```bash
   conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
   ```
   Or:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
2. Verify: `python -c "import torch; print(torch.cuda.is_available())"`
3. CPU training works fine, just slower

## XGBoost GPU Falls Back to CPU

**Message**: `XGBoost GPU training failed ... Retrying on CPU.`

**Solution**:
1. This is expected if CUDA-enabled XGBoost cannot access the GPU.
2. Keep `model.gpu_fallback: true` for robust runs.
3. If you want to force CPU, set:
   ```yaml
   model:
     device: cpu
   ```
4. If you want to force GPU and fail loudly, set:
   ```yaml
   model:
     device: cuda
     gpu_fallback: false
   ```

## Shape Mismatch

**Error**: `ValueError: Expected I/Q data shape (N, sample_length, 2), got ...`

**Solution**:
1. Verify your dataset format matches RadioML 2018.01A
2. Run `rf-sentinel inspect-data` to check shapes
3. The expected shape is `(N, 1024, 2)`

## MLflow UI Not Opening

**Issue**: `mlflow ui` command doesn't work

**Solution**:
1. Ensure MLflow is installed: `pip install mlflow`
2. Run from the project root: `mlflow ui --backend-store-uri mlruns`
3. Open `http://127.0.0.1:5000` in your browser
4. If port 5000 is in use: `mlflow ui --port 5001 --backend-store-uri mlruns`

## Streamlit Cannot Find Checkpoint

**Error**: Streamlit shows "Checkpoint not found"

**Solution**:
1. Train a model first, for example: `rf-sentinel train --config configs/cnn1d_baseline.yaml`
2. Verify the matching checkpoint exists, for example `artifacts/checkpoints/cnn1d_best.pt`, `artifacts/checkpoints/resnet1d_best.pt`, or `artifacts/checkpoints/tcn1d_best.pt`
3. Update the checkpoint path in the Streamlit sidebar

## Unsupported Model Type

**Error**: `Unsupported waveform model type ...`

**Solution**:
1. For raw I/Q PyTorch training, use `model.type: cnn1d`, `resnet1d`, or `tcn1d`.
2. For XGBoost, use the separate command: `rf-sentinel train-xgboost --config configs/xgboost_features.yaml`.
3. Check spelling in the YAML config.

## Import Errors

**Error**: `ModuleNotFoundError: No module named 'rf_sentinel'`

**Solution**:
1. Install in editable mode: `pip install -e .`
2. Make sure your virtual environment is activated
3. Verify: `python -c "import rf_sentinel; print(rf_sentinel.__version__)"`

## XGBoost Not Installed

**Error**: `XGBoost is not installed`

**Solution**:
```bash
pip install xgboost
# or
pip install -e ".[xgboost]"
```
