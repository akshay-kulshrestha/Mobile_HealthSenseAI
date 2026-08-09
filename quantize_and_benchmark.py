import os
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# File paths
FP32_MODEL_PATH = "HealthSenseNet.onnx"
CLEAN_MODEL_PATH = "HealthSenseNet_clean.onnx"
INT8_MODEL_PATH = "HealthSenseNet_quantized_int8.onnx"


def clean_onnx_shapes(input_path, output_path):
    """Strip stale intermediate shape metadata (value_info) from PyTorch ONNX export."""
    print("🧹 Cleaning intermediate shape metadata...")
    model = onnx.load(input_path)
    model.graph.ClearField("value_info")
    onnx.save(model, output_path)


def quantize_model():
    """Quantize FP32 ONNX model to INT8."""
    # 1. Clean graph shape metadata
    clean_onnx_shapes(FP32_MODEL_PATH, CLEAN_MODEL_PATH)

    # 2. Perform INT8 Dynamic Quantization
    print("⚡ Starting INT8 Dynamic Quantization...")
    quantize_dynamic(
        model_input=CLEAN_MODEL_PATH,
        model_output=INT8_MODEL_PATH,
        weight_type=QuantType.QInt8,
    )

    # 3. Clean up temporary clean model artifact
    if os.path.exists(CLEAN_MODEL_PATH):
        os.remove(CLEAN_MODEL_PATH)

    print(f"✅ Quantized model saved to: {INT8_MODEL_PATH}")


def benchmark_model(model_path, sample_input, num_runs=1000, warmup=100):
    """Run latency benchmarks for an ONNX model."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Warm-up runs to stabilize CPU cache & execution
    for _ in range(warmup):
        session.run(None, {input_name: sample_input})

    # Benchmark loop
    start_time = time.perf_counter()
    for _ in range(num_runs):
        outputs = session.run(None, {input_name: sample_input})
    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000
    file_size_kb = os.path.getsize(model_path) / 1024

    return avg_latency_ms, file_size_kb, outputs[0]


def main():
    if not os.path.exists(FP32_MODEL_PATH):
        print(f"❌ Error: {FP32_MODEL_PATH} not found in working directory.")
        return

    # 1. Perform Quantization
    try:
        quantize_model()
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        return

    # 2. Extract dynamic input dimensions
    session_temp = ort.InferenceSession(
        FP32_MODEL_PATH, providers=["CPUExecutionProvider"]
    )
    input_shape = [
        dim if isinstance(dim, int) else 1
        for dim in session_temp.get_inputs()[0].shape
    ]
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    print("\n⏱️  Benchmarking Models (1,000 runs on CPU)...")

    # 3. Benchmark FP32
    fp32_latency, fp32_size, fp32_out = benchmark_model(
        FP32_MODEL_PATH, dummy_input
    )

    # 4. Benchmark INT8
    int8_latency, int8_size, int8_out = benchmark_model(
        INT8_MODEL_PATH, dummy_input
    )

    # 5. Output Summary
    size_reduction = ((fp32_size - int8_size) / fp32_size) * 100
    speedup = fp32_latency / int8_latency if int8_latency > 0 else 1.0
    mae = float(np.mean(np.abs(fp32_out - int8_out)))

    print("\n" + "=" * 50)
    print("📊 PERFORMANCE COMPARISON SUMMARY")
    print("=" * 50)
    print(f"FP32 Model Size     : {fp32_size:.2f} KB")
    print(f"INT8 Model Size     : {int8_size:.2f} KB ({size_reduction:.1f}% smaller)")
    print(f"FP32 Avg Latency    : {fp32_latency:.4f} ms / inference")
    print(f"INT8 Avg Latency    : {int8_latency:.4f} ms / inference")
    print(f"Latency Speedup     : {speedup:.2f}x")
    print(f"Mean Abs Error (MAE): {mae:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()