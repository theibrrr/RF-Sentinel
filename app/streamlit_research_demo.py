"""RF-Sentinel Streamlit Research Demo.

Minimal single-sample research demo for inspecting predictions.

Usage:
    streamlit run app/streamlit_research_demo.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Ensure project is importable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

st.set_page_config(page_title="RF-Sentinel Research Demo", page_icon="RF", layout="wide")

st.title("RF-Sentinel Research Demo")
st.markdown("**Single-sample modulation classification from raw I/Q data**")
st.markdown("---")


@st.cache_resource
def load_predictor(checkpoint_path, device, threshold, top_k):
    from rf_sentinel.inference.predictor import RFPredictor

    return RFPredictor(
        checkpoint_path=checkpoint_path,
        device_cfg=device,
        confidence_threshold=threshold,
        top_k=top_k,
    )


@st.cache_data
def load_dataset_sample(dataset_path, x_key, y_key, snr_key, index):
    import h5py

    with h5py.File(dataset_path, "r") as f:
        sample = f[x_key][index]
        y_raw = f[y_key][index]
        snr_val = float(f[snr_key][index].flatten()[0]) if snr_key in f else None

    label_idx = int(np.argmax(y_raw)) if y_raw.ndim >= 1 and len(y_raw) > 1 else int(y_raw)

    return sample.astype(np.float32), label_idx, snr_val


# Sidebar configuration
st.sidebar.header("Configuration")

checkpoint_path = st.sidebar.text_input(
    "Checkpoint path",
    value="artifacts/checkpoints/cnn1d_best.pt",
)

dataset_path = st.sidebar.text_input(
    "Dataset path (HDF5)",
    value="data/raw/GOLD_XYZ_OSC.0001_1024.hdf5",
)

x_key = st.sidebar.text_input("X key", value="X")
y_key = st.sidebar.text_input("Y key", value="Y")
snr_key = st.sidebar.text_input("SNR key", value="Z")

device = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.70, 0.05)
top_k = st.sidebar.slider("Top-K predictions", 1, 10, 3)

sample_index = st.sidebar.number_input("Sample index", min_value=0, value=0, step=1)

# Main content
if st.sidebar.button("Run Prediction", type="primary"):
    if not Path(checkpoint_path).exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
    elif not Path(dataset_path).exists():
        st.error(f"Dataset not found: {dataset_path}")
    else:
        try:
            # Load sample
            sample, true_label_idx, snr_val = load_dataset_sample(
                dataset_path,
                x_key,
                y_key,
                snr_key,
                sample_index,
            )

            # Load predictor
            predictor = load_predictor(checkpoint_path, device, threshold, top_k)

            # Get true label name
            true_label = predictor.index_to_label.get(true_label_idx, f"Class_{true_label_idx}")

            # Display sample info
            col1, col2, col3 = st.columns(3)
            col1.metric("True Label", true_label)
            col2.metric("Sample Index", sample_index)
            if snr_val is not None:
                col3.metric("SNR (dB)", f"{snr_val:.0f}")

            st.markdown("---")

            # Plot waveforms
            st.subheader("I/Q Waveforms")
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=1, cols=2, subplot_titles=("I/Q Channels", "Amplitude"))

            I_data = sample[:, 0]
            Q_data = sample[:, 1]
            amplitude = np.sqrt(I_data**2 + Q_data**2)
            time_axis = np.arange(len(I_data))

            fig.add_trace(
                go.Scatter(x=time_axis, y=I_data, name="I", line=dict(color="#2196F3")),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=time_axis, y=Q_data, name="Q", line=dict(color="#FF5722")),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=time_axis, y=amplitude, name="Amplitude", line=dict(color="#4CAF50")),
                row=1,
                col=2,
            )

            fig.update_layout(height=350, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Run prediction
            st.subheader("Prediction Result")
            result = predictor.predict(sample)

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Class", result["prediction"])
            col2.metric("Confidence", f"{result['confidence']:.4f}")

            if result["decision"] == "accepted":
                col3.metric("Decision", "Accepted")
            else:
                col3.metric("Decision", "Uncertain")

            # Correct / incorrect
            if result["prediction"] == true_label:
                st.success("Prediction is **correct**.")
            else:
                st.error(f"Prediction is **incorrect**. True label: {true_label}")

            # Top-k
            st.subheader(f"Top-{top_k} Predictions")
            for i, pred in enumerate(result["top_k"], 1):
                st.write(f"**{i}. {pred['class']}** - {pred['probability']:.4f}")
                st.progress(pred["probability"])

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

else:
    st.info("Configure settings in the sidebar and click **Run Prediction** to start.")
    st.markdown("""
    ### How to use
    1. Set the **checkpoint path** to your trained model
    2. Set the **dataset path** to the RadioML HDF5 file
    3. Choose a **sample index**
    4. Click **Run Prediction**

    ### Requirements
    - A trained model checkpoint (run `rf-sentinel train` first)
    - The RadioML 2018.01A dataset
    """)
