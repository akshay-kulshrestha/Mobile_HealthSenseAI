import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Synthetic Mobile/Wearable Sensor Generator
def generate_synthetic_sensor_data(n_samples=2000, seq_len=100, n_features=7):
    X = np.zeros((n_samples, seq_len, n_features))
    y = np.random.choice([0, 1, 2, 3], size=n_samples)
    time = np.linspace(0, 10, seq_len)

    for i in range(n_samples):
        label = y[i]
        freq = 0.5 if label == 0 else (1.5 if label == 1 else (3.0 if label == 2 else 4.5))
        for f in range(6):
            X[i, :, f] = np.sin(2 * np.pi * freq * time + f) + np.random.normal(0, 0.2, seq_len)
        base_hr = 60 if label == 0 else (90 if label == 1 else (140 if label == 2 else 120))
        X[i, :, 6] = base_hr + np.random.normal(0, 3.0, seq_len)
    return X, y

# 2. PyTorch Dataset with Noise Injection
class MobileSensorDataset(Dataset):
    def __init__(self, data, labels, inject_noise=False, noise_level=0.1):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.inject_noise = inject_noise
        self.noise_level = noise_level

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.inject_noise:
            x = x + torch.randn_like(x) * self.noise_level
            mask = (torch.rand_like(x) > 0.05).float()
            x = x * mask
        return x.transpose(0, 1), y  # (Channels, Seq_Len)

# 3. Trustworthy Deep Neural Network Architecture
class HealthSenseNet(nn.Module):
    def __init__(self, input_channels=7, num_classes=4, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)

if __name__ == "__main__":
    print("[1/3] Generating Synthetic Sensor Data...")
    X, y = generate_synthetic_sensor_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_loader = DataLoader(MobileSensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(MobileSensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    model = HealthSenseNet()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("[2/3] Training HealthSenseNet (1D-CNN + LSTM)...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   Epoch {epoch+1}/5 | Training Loss: {total_loss/len(train_loader):.4f}")

    # Final Model Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds = model(X_b).argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)
    print(f"\n[3/3] Target Evaluation Accuracy: {correct/total * 100:.2f}%\n")