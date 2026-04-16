import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models import BodyTransformer

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# DATASET
# =========================
class BodyDataset(Dataset):
    def __init__(self, csv_path):
        print("📂 Loading CSV:", csv_path)

        df = pd.read_csv(csv_path)

        if df.shape[0] == 0:
            raise ValueError("❌ CSV is empty")

        # =========================
        # INPUT (264 features)
        # =========================
        self.X = df.iloc[:, :264].values.astype(np.float32)

        # =========================
        # OUTPUT (6 measurements)
        # =========================
        self.Y = df.iloc[:, 264:].values.astype(np.float32)

        # =========================
        # NORMALIZATION (IMPORTANT)
        # =========================
        self.X = self.X / 1.0   # already normalized pose features

        # scale body measurements (cm → normalized)
        self.Y = self.Y / np.array([200, 200, 200, 200, 200, 200], dtype=np.float32)

        # store mean for later denormalization (optional production use)
        self.y_scale = np.array([200, 200, 200, 200, 200, 200], dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x, y


# =========================
# TRAIN
# =========================
def train():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "dataset.csv")

    print("🚀 Loading dataset...")
    dataset = BodyDataset(csv_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("🧠 Initializing model...")
    model = BodyTransformer().to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    print("🔥 Training started...")

    for epoch in range(50):
        model.train()
        total_loss = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Transformer expects sequence dimension
            x = x.unsqueeze(1)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/50] Loss: {total_loss:.4f}")

    # =========================
    # SAVE MODEL
    # =========================
    save_path = os.path.join(BASE_DIR, "body_ai_model.pth")
    torch.save(model.state_dict(), save_path)

    print("✅ Model saved at:", save_path)


if __name__ == "__main__":
    train()