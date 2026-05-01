# Experiment Tracking with MLflow

RF-Sentinel uses [MLflow](https://mlflow.org/) for experiment tracking.

## Configuration

MLflow is configured via the `mlflow` section in your YAML config:

```yaml
mlflow:
  enabled: true
  tracking_uri: mlruns
  experiment_name: rf-sentinel-waveform-models
  run_name: resnet1d-100-epochs
```

Set `enabled: false` to disable tracking (e.g., for quick tests).

## What Gets Logged

### Parameters
- Model type, base channels, dropout
- ResNet block counts or TCN dilation settings when present in the config
- Learning rate, batch size, epochs, optimizer
- Normalization method
- Split seed
- Max samples / subset fraction

### Metrics (Training)
- `train_loss` — per epoch
- `val_loss` — per epoch
- `val_accuracy` — per epoch
- `val_macro_f1` — per epoch

### Metrics (Evaluation)
- `test_accuracy`
- `macro_f1`, `weighted_f1`
- `low_snr_accuracy`, `high_snr_accuracy`
- `accepted_ratio`, `uncertain_ratio`

### Artifacts
- Config YAML copy
- Confusion matrix figure
- Accuracy vs SNR figure
- Classification report CSV
- Evaluation metrics JSON
- Error analysis markdown
- Model checkpoint (`cnn1d_best.pt`, `resnet1d_best.pt`, `tcn1d_best.pt`, etc.)

## Viewing Results

Start the MLflow UI:
```bash
mlflow ui --backend-store-uri mlruns
```

Then open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Comparing Runs

In the MLflow UI you can:
1. Select multiple runs in the experiment view
2. Click "Compare" to see side-by-side metrics
3. Use the chart view to plot metrics across runs
4. Filter runs by parameters or metrics

For clean CNN1D/ResNet1D/TCN1D comparisons, keep the same `experiment_name`
across the configs and give each training run a descriptive `run_name`.
