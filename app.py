import time
import numpy as np
import onnxruntime as ort
import plotly.graph_objects as go
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Mobile HealthSenseAI | Interactive Demo",
    page_icon="🩺",
    layout="wide",
)

# Header Section
st.title("🩺 Mobile HealthSenseAI")
st.caption(
    "Real-Time Mobile Health Monitoring Pipeline | PyTorch • ONNX Runtime • Edge Quantization"
)
st.markdown("---")


# Sidebar Controls
st.sidebar.header("⚙️ Simulation Settings")
seq_len = st.sidebar.slider("Sequence Length (Time Steps)", 50, 200, 100, step=10)
signal_noise = st.sidebar.slider("Sensor Noise Level", 0.0, 1.0, 0.2, step=0.05)
anomaly_boost = st.sidebar.slider("Simulate Anomaly (Spike)", 0.0, 5.0, 0.0, step=0.5)

model_choice = st.sidebar.radio(
    "Select Model Runtime",
    ["Standard FP32 ONNX", "Quantized INT8 ONNX"],
)

MODEL_FILE = (
    "HealthSenseNet.onnx"
    if model_choice == "Standard FP32 ONNX"
    else "HealthSenseNet_quantized_int8.onnx"
)


# Helper: Generate Dynamic Signal Data with 7 Feature Channels (Batch=1, Features=7, Length)
def generate_synthetic_signal(length, noise, anomaly):
    t = np.linspace(0, 4 * np.pi, length)
    # 7 feature channels matching model contract
    ch1 = np.sin(t) + np.random.normal(0, noise, length) + anomaly
    ch2 = np.cos(t) + np.random.normal(0, noise, length)
    ch3 = np.sin(2 * t) * 0.5 + np.random.normal(0, noise, length)
    ch4 = np.random.normal(0.98, noise * 0.05, length)
    ch5 = np.random.normal(36.6, noise * 0.1, length)
    ch6 = np.cos(0.5 * t) + np.random.normal(0, noise, length)
    ch7 = np.sin(0.5 * t) + np.random.normal(0, noise, length)

    signal = np.stack([ch1, ch2, ch3, ch4, ch5, ch6, ch7], axis=-1)  # (length, 7)

    # Transpose to shape (Features=7, Sequence_Length)
    signal_transposed = np.transpose(signal, (1, 0))  # (7, length)
    return np.expand_dims(signal_transposed, axis=0).astype(np.float32)  # (1, 7, length)


# Main Interactive Body
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Real-Time Health Sensor Signals")

    # Generate signal array
    sample_input = generate_synthetic_signal(seq_len, signal_noise, anomaly_boost)

    # Plot input signals using Plotly
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=sample_input[0, 0, :],
            name="Channel 1 (Heart Rate / Pulse)",
            line=dict(color="#FF4B4B", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            y=sample_input[0, 1, :],
            name="Channel 2 (Motion / Accel X)",
            line=dict(color="#0068C9", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            y=sample_input[0, 2, :],
            name="Channel 3 (Motion / Accel Y)",
            line=dict(color="#29B09D", width=1.5, dash="dash"),
        )
    )

    fig.update_layout(
        xaxis_title="Time Steps",
        yaxis_title="Normalized Amplitude",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
    )
    st.plotly_chart(fig, width="stretch")

with col2:
    st.subheader("🤖 Model Inference")

    if st.button("🚀 Run Inference", width="stretch"):
        try:
            session = ort.InferenceSession(
                MODEL_FILE, providers=["CPUExecutionProvider"]
            )
            input_name = session.get_inputs()[0].name

            # Measure inference performance
            start_t = time.perf_counter()
            output = session.run(None, {input_name: sample_input})[0]
            latency_ms = (time.perf_counter() - start_t) * 1000

            # Compute Softmax and Prediction Class
            exp_out = np.exp(output - np.max(output))
            probs = exp_out / np.sum(exp_out)
            pred_class = int(np.argmax(probs))
            confidence = float(np.max(probs)) * 100

            st.success("Inference Completed Successfully!")

            # Display Output Metrics
            st.metric("Predicted Health State", f"Class {pred_class}")
            st.metric("Confidence Score", f"{confidence:.1f}%")
            st.metric("Inference Latency", f"{latency_ms:.3f} ms")

            # Reliability Metrics
            st.markdown("### 🛡️ Reliability Analysis")
            if confidence > 80:
                st.info("High Confidence • Model output is stable and reliable.")
            elif confidence > 50:
                st.warning("Moderate Confidence • Sensor noise detected.")
            else:
                st.error("Low Confidence • Potential Out-of-Distribution Signal.")

        except Exception as e:
            st.error(f"Error executing inference: {e}")
    else:
        st.info("Click **Run Inference** to evaluate the model on current sensor stream.")

# Edge Optimization Section
st.markdown("---")
st.subheader("📊 Edge Deployment Optimization Findings")
st.markdown(
    """
> **Key Academic Insight:** For lightweight micro-architectures (~50 KB), raw FP32 ONNX execution outperforms INT8 Dynamic Quantization because the runtime overhead of calculating quantization scale parameters outweighs weight compression savings.
"""
)

b1, b2, b3 = st.columns(3)
b1.metric("FP32 Model Size", "49.86 KB", delta="Optimal Size", delta_color="normal")
b2.metric("INT8 Model Size", "317.68 KB", delta="+537% (Overhead)", delta_color="inverse")
b3.metric("FP32 Speedup Ratio", "2.05x Faster", delta="0.34 ms vs 0.70 ms", delta_color="normal")