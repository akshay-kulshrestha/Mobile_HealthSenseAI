import torch
from health_sense_ai import HealthSenseNet, generate_synthetic_sensor_data

def test_sensor_data_shape():
    X, y = generate_synthetic_sensor_data(n_samples=20)
    assert X.shape == (20, 100, 7)
    assert len(y) == 20

def test_model_forward_shape():
    model = HealthSenseNet()
    tensor = torch.randn(4, 7, 100)
    out = model(tensor)
    assert out.shape == (4, 4)