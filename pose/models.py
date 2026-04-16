import torch
import torch.nn as nn
import torch.nn.functional as F

class BodyTransformer(nn.Module):
    def __init__(self, input_size=264, d_model=256, output_size=6):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        # 🔥 Case 1: agar shape (1,1,280) hai → squeeze karo
        if len(x.shape) == 3:
            x = x.squeeze(1)   # (1,1,280) → (1,280)

        # 🔥 Fix feature size mismatch
        EXPECTED_FEATURES = 264

        if x.shape[1] > EXPECTED_FEATURES:
            x = x[:, :EXPECTED_FEATURES]
        elif x.shape[1] < EXPECTED_FEATURES:
            pad = EXPECTED_FEATURES - x.shape[1]
            x = F.pad(x, (0, pad))

        # 🔥 Input projection
        x = self.input_proj(x)      # (batch, 256)

        # 🔥 Transformer ke liye sequence dimension add karo
        x = x.unsqueeze(1)          # (batch, 1, 256)

        x = self.transformer(x)

        x = x[:, -1, :]             # (batch, 256)

        return self.head(x)