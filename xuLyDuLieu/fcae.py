# xuLyDuLieu/fcae.py  
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


class FCAE(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.dec = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_size)
        )

    def forward(self, x):
        enc = self.enc(x)
        dec = self.dec(enc)
        return enc, dec

    def encode(self, x):
        return self.enc(x)


class FeaturesDataset(Dataset):
    def __init__(self, samples):
        self.samples = torch.FloatTensor(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def extract_features(price_history, past_days=32, epochs=100, batch_size=128, plot_loss=True):
    """
    Hàm chính: nhận price_history → trả về features đã lọc nhiễu
    """
    prices = price_history['close'].values  # sửa thành 'close' cho chắc

    samples = []
    for i in range(len(prices) - past_days + 1):
        window = prices[i:i + past_days]
        samples.append(window)

    samples = np.array(samples, dtype=np.float32)

    # Chuẩn hóa min-max từng cửa sổ
    samples_min = samples.min(axis=1, keepdims=True)
    samples_max = samples.max(axis=1, keepdims=True)
    samples_norm = (samples - samples_min) / (samples_max - samples_min + 1e-8)

    dataset = FeaturesDataset(samples_norm)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCAE(input_size=past_days).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.HuberLoss()

    model.train()
    losses = []

    for _ in tqdm(range(epochs), desc="Training FCAE"):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            _, recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(dataloader))

    # Encode
    model.eval()
    with torch.no_grad():
        all_data = torch.FloatTensor(samples_norm).to(device)
        encoded = model.encode(all_data).cpu().numpy()

    if plot_loss:
        plt.figure(figsize=(10, 4))
        plt.plot(losses)
        plt.title(f"FCAE Loss - past_days={past_days}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

    print(f"Extract features hoàn tất! Shape: {encoded.shape}")
    return encoded, model, losses