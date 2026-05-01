# Model Cards: RF-Sentinel Waveform Models

RF-Sentinel supports three PyTorch models that consume the same raw I/Q input:
`(batch, 2, 1024)` after per-sample normalization.

## Shared Task

- **Task**: RF modulation classification
- **Input**: raw I/Q waveform, 2 channels, 1024 time steps
- **Output**: logits over 24 RadioML 2018.01A modulation classes
- **Default normalization**: per-sample RMS normalization
- **Training split**: 70% train / 15% validation / 15% test
- **Stratification**: label + SNR when feasible

## CNN1D Baseline

Compact convolutional baseline:

```text
Input (batch, 2, 1024)
Conv1d -> BatchNorm -> ReLU -> MaxPool
Conv1d -> BatchNorm -> ReLU -> MaxPool
Conv1d -> BatchNorm -> ReLU -> AdaptiveAvgPool
Dropout -> Linear(num_classes)
```

Use it when you want the simplest deep baseline and fastest iteration.

Config examples:

```bash
rf-sentinel train --config configs/cnn1d_baseline.yaml
rf-sentinel train --config configs/cnn1d_quick_test.yaml
```

## ResNet1D

Residual 1D CNN for deeper raw-waveform learning:

```text
Input (batch, 2, 1024)
Stem Conv1d -> BatchNorm -> ReLU
Residual stage 1
Residual stage 2 with downsampling
Residual stage 3 with downsampling
AdaptiveAvgPool -> Dropout -> Linear(num_classes)
```

Use it when CNN1D underfits or when you want a stronger raw-I/Q architecture
without changing the data representation.

Key config fields:

```yaml
model:
  type: resnet1d
  base_channels: 64
  blocks_per_stage: [2, 2, 2]
```

## TCN1D / Dilated CNN

Dilated temporal convolution model:

```text
Input (batch, 2, 1024)
TemporalBlock dilation=1
TemporalBlock dilation=2
TemporalBlock dilation=4
TemporalBlock dilation=8
...
AdaptiveAvgPool -> Linear(num_classes)
```

Use it when longer temporal context matters. Dilations increase the receptive
field without requiring recurrent layers or attention.

Key config fields:

```yaml
model:
  type: tcn1d
  base_channels: 64
  num_blocks: 5
  kernel_size: 5
  dilation_base: 2
```

## Metrics

All waveform models use the same evaluation pipeline:

- Overall accuracy
- Macro F1 and weighted F1
- Per-class precision/recall/F1
- Confusion matrix
- Accuracy vs. SNR
- Low-SNR vs. high-SNR accuracy
- Confidence-aware accepted/uncertain metrics
- Error analysis markdown report

## Known Limitations

1. **Low-SNR performance**: Classification accuracy degrades significantly at low SNR, which is expected for RF modulation recognition.
2. **Similar modulation families**: QPSK/OQPSK, PSK variants, APSK variants, and QAM variants can remain difficult to separate.
3. **Synthetic benchmark data**: The models are trained and evaluated on RadioML simulated data, not validated live on SDR hardware.
4. **Fixed sample length**: Checkpoints expect the sample length saved in checkpoint metadata, usually 1024.
5. **No real-time deployment path in v1**: Streaming, ONNX export, FastAPI serving, and SDR integrations are future work.

## Model Selection Guidance

- Start with **CNN1D** for a baseline.
- Try **ResNet1D** when you need a stronger raw-waveform model.
- Try **TCN1D** when you want a larger temporal receptive field with a compact convolutional design.
- Use **XGBoost** as a classical feature-based baseline, not as a replacement for raw waveform learning.
