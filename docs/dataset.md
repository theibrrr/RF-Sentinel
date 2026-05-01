# Dataset Documentation

## DeepSig RadioML 2018.01A

RF-Sentinel is designed to work with the **DeepSig RadioML 2018.01A** dataset, a widely used benchmark for automatic modulation classification (AMC) research.

### Overview

The dataset contains synthetically generated RF signals with various modulation types at different signal-to-noise ratios (SNR). Each sample consists of raw in-phase (I) and quadrature (Q) components.

### Expected File

The dataset is typically distributed as an HDF5 file:
```
GOLD_XYZ_OSC.0001_1024.hdf5
```

### HDF5 Structure

| Key | Shape | Description |
|-----|-------|-------------|
| `X` | `(N, 1024, 2)` | Raw I/Q samples. Dimension 0 = samples, dimension 1 = time steps (1024), dimension 2 = channels (I and Q) |
| `Y` | `(N, 24)` | One-hot encoded modulation labels (24 classes) |
| `Z` | `(N, 1)` | SNR values in dB |

### Modulation Classes (24)

| Digital | Analog |
|---------|--------|
| OOK, 4ASK, 8ASK | AM-SSB-WC, AM-SSB-SC |
| BPSK, QPSK, 8PSK, 16PSK, 32PSK | AM-DSB-WC, AM-DSB-SC |
| 16APSK, 32APSK, 64APSK, 128APSK | FM |
| 16QAM, 32QAM, 64QAM, 128QAM, 256QAM | GMSK, OQPSK |

### SNR Range

- **Range**: -20 dB to +30 dB
- **Step**: 2 dB
- **Total SNR values**: 26

### I/Q Data

Each sample contains 1024 time steps of complex baseband signal:
- **I (In-phase)**: Real component of the complex signal
- **Q (Quadrature)**: Imaginary component of the complex signal

The complex representation: `s(t) = I(t) + jQ(t)`

### Large-File Handling

RadioML 2018.01A is large. The common `X` dataset is about 20 GB as float32.
RF-Sentinel decodes labels and SNR metadata first, then:

- reads only selected samples when `data.max_samples` or `data.subset_fraction` is set
- keeps the full `X` array on disk and uses lazy HDF5 reads for full CNN runs
- avoids creating a full preprocessed copy of the complete dataset in RAM

### Dataset Not Included

The dataset is **not** included in this repository. You must download it separately and place it at:
```
data/raw/GOLD_XYZ_OSC.0001_1024.hdf5
```

Or configure a custom path in your YAML config:
```yaml
data:
  dataset_path: /path/to/your/dataset.hdf5
```

### Inspection Command

After placing the dataset:
```bash
rf-sentinel inspect-data --config configs/cnn1d_baseline.yaml
```

This generates:
- `reports/dataset_summary.json`
- `reports/dataset_summary.md`
