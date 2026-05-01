"""
models.py — BodyTransformer with gender/age-aware architecture.

Input feature vector size breakdown (per fused_features):
  10  geometric features   (fused_geo)
   4  gender one-hot       (gender_fused)
   4  front visibility
 132  front raw landmarks  (33 × 4)
   4  side visibility
 132  side raw landmarks   (33 × 4)
  ──────────────────────────────────
 286  TOTAL INPUT FEATURES

Output: 6 measurements [chest, waist, hip, shoulder, sleeve, inseam]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_SIZE  = 286
OUTPUT_SIZE = 6
D_MODEL     = 256


class BodyTransformer(nn.Module):
    """
    Transformer-based body measurement predictor.

    Architecture improvements over v1:
    - Correct input size (286) matching updated ai_utils feature vector.
    - Gender-aware gating: a learned gate scales transformer output based
      on the gender one-hot embedding, letting the model learn different
      measurement patterns for male / female / kid.
    - Deeper head (3 → 4 linear layers) with dropout for regularization.
    - LayerNorm on input projection output.
    """

    def __init__(
        self,
        input_size:  int = INPUT_SIZE,
        d_model:     int = D_MODEL,
        output_size: int = OUTPUT_SIZE,
        nhead:       int = 8,
        num_layers:  int = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
        )

        # Gender gate — projects gender one-hot (4-dim) to d_model scale factors
        self.gender_gate = nn.Sequential(
            nn.Linear(4, d_model),
            nn.Sigmoid(),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, input_size) or (batch, input_size)
        Returns:
            Tensor of shape (batch, output_size)
        """
        # Handle (batch, 1, N) → (batch, N)
        if x.dim() == 3:
            x = x.squeeze(1)

        # Dynamic size fix: pad or trim to expected input_size
        if x.shape[1] > self.input_size:
            x = x[:, :self.input_size]
        elif x.shape[1] < self.input_size:
            pad_size = self.input_size - x.shape[1]
            x = F.pad(x, (0, pad_size))

        # Extract gender one-hot (indices 10:14 in fused feature vector)
        gender_vec = x[:, 10:14]   # [male, female, kid, confidence]

        # Input projection with gender gating
        projected  = self.input_proj(x)          # (batch, d_model)
        gate       = self.gender_gate(gender_vec) # (batch, d_model)
        gated      = projected * gate             # element-wise scale

        # Transformer expects (batch, seq_len, d_model)
        seq = gated.unsqueeze(1)                  # (batch, 1, d_model)
        out = self.transformer(seq)               # (batch, 1, d_model)
        out = out[:, -1, :]                       # (batch, d_model)

        return self.head(out)                     # (batch, output_size)


# ---------------------------------------------------------------------------
# Training utils (only needed for training, not inference)
# ---------------------------------------------------------------------------
class MeasurementLoss(nn.Module):
    """
    Weighted MSE loss. Chest and waist are weighted higher because
    they are harder to estimate and most critical for tailoring.
    """

    def __init__(self):
        super().__init__()
        # [chest, waist, hip, shoulder, sleeve, inseam]
        self.weights = torch.tensor(
            [1.5, 1.5, 1.2, 1.0, 1.0, 1.0], dtype=torch.float32
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.weights.to(pred.device)
        return (w * (pred - target) ** 2).mean()