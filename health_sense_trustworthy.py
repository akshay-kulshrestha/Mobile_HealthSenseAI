import torch
import numpy as np
from health_sense_ai import HealthSenseNet

# Monte Carlo Dropout Uncertainty Estimator
def estimate_uncertainty(model, input_tensor, num_samples=20):
    model.train()  # Keep dropout layers active during inference
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            probs = torch.softmax(model(input_tensor), dim=-1)
            predictions.append(probs.numpy())
    preds_arr = np.array(predictions)
    mean_probs = preds_arr.mean(axis=0)
    uncertainty = preds_arr.var(axis=0).mean(axis=-1)
    return mean_probs.argmax(axis=1), uncertainty

# ONNX Mobile Export Engine
def export_onnx(model, filename="HealthSenseNet.onnx"):
    model.eval()
    dummy_input = torch.randn(1, 7, 100)
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        input_names=['sensor_input'],
        output_names=['health_prediction'],
        dynamic_axes={'sensor_input': {0: 'batch_size'}, 'health_prediction': {0: 'batch_size'}}
    )
    print(f"[✓] Successfully exported ONNX model to '{filename}'")

if __name__ == "__main__":
    print("--- Running Trustworthy AI & Mobile Export Pipeline ---")
    model = HealthSenseNet()
    sample_batch = torch.randn(4, 7, 100)

    preds, uncertainty = estimate_uncertainty(model, sample_batch)
    print("\nMonte Carlo Uncertainty Outputs:")
    for idx in range(len(preds)):
        print(f" Sample {idx+1}: Predicted Health State = {preds[idx]} | Epistemic Uncertainty = {uncertainty[idx]:.5f}")

    print("\nExporting for Mobile Deployment:")
    export_onnx(model)