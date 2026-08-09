# Mobile HealthSenseAI 🩺⚡

> An end-to-end lightweight deep learning pipeline for real-time mobile health monitoring, featuring ONNX optimization for edge deployment and reliable automated test coverage.

---

## 📌 Project Overview

**Mobile HealthSenseAI** is designed to process sequential health signals on resource-constrained mobile and edge devices. By leveraging PyTorch and exporting optimized models to the **ONNX (Open Neural Network Exchange)** runtime, this project bridges the gap between deep learning model development and low-latency on-device inference.

### Key Features
* **Baseline Architecture:** Hybrid deep learning approach combining 1D-CNN feature extraction with sequential LSTM processing for temporal health data.
* **Trustworthy AI Mechanics:** Integrated evaluation metrics for model confidence, uncertainty, and baseline robustness checks.
* **Edge Deployment Ready:** Automated export to `.onnx` format for fast execution on mobile platforms (Android/iOS/Embedded).
* **Automated Testing:** Fully unit-tested test suite using `pytest` to guarantee model output consistency and contract validity.

---

## 🛠️ Project Structure

```text
Mobile_HealthSenseAI/
├── health_sense_ai.py           # Core model architecture & training pipeline
├── health_sense_trustworthy.py  # Model evaluation, uncertainty, & trustworthiness checks
├── test_health_sense.py         # Pytest suite for model shape & inference verification
├── HealthSenseNet.onnx          # Exported ONNX model ready for mobile deployment
├── .gitignore                   # Excluded virtual environments and cache files
└── README.md                    # Project documentation
