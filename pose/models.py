"""
models.py — BodyTransformer with gender/age-aware architecture.

FIX: INPUT_SIZE corrected to match ai_utils.py feature vector.

Feature vector breakdown (per fused_features from fuse_features()):
  16  geometric features   (N_GEO=16, fused_geo)
   4  gender one-hot       (gender_fused)
   4  front visibility
 132  front raw landmarks  (33 × 4)
   4  side visibility
 132  side raw landmarks   (33 × 4)
  ──────────────────────────────────
 292  TOTAL

Note: models.py original had INPUT_SIZE=286 (based on N_GEO=10).
      Now N_GEO=16 → INPUT_SIZE = 16+4+4+132+4+132 = 292.
      The pad/trim in forward() still works as safety net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Derived from ai_utils.py: N_GEO=16, gender=4, vis=4, lm=132, ×2 views
INPUT_SIZE  = 292   # FIX: was 286 (N_GEO=10), now 292 (N_GEO=16)
OUTPUT_SIZE = 6
D_MODEL     = 256


class BodyTransformer(nn.Module):
    """
    Transformer-based body measurement predictor.

    Input:  (batch, 1, INPUT_SIZE) or (batch, INPUT_SIZE)
    Output: (batch, 6)  → [chest, waist, hip, shoulder, sleeve, inseam] normalized 0-1

    Gender-aware gating: learns different measurement patterns per gender.
    Gender one-hot is at fused_features[16:20] (after N_GEO=16 geometric).
    """

    def __init__(
        self,
        input_size:  int   = INPUT_SIZE,
        d_model:     int   = D_MODEL,
        output_size: int   = OUTPUT_SIZE,
        nhead:       int   = 8,
        num_layers:  int   = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
        )

        # Gender gate uses indices 16:20 (N_GEO=16 → gender starts at 16)
        self.gender_gate = nn.Sequential(
            nn.Linear(4, d_model),
            nn.Sigmoid(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

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
        if x.dim() == 3:
            x = x.squeeze(1)

        # Pad or trim to expected input_size (safety net for version mismatches)
        if x.shape[1] > self.input_size:
            x = x[:, :self.input_size]
        elif x.shape[1] < self.input_size:
            x = F.pad(x, (0, self.input_size - x.shape[1]))

        # Gender one-hot at indices 16:20 (N_GEO=16)
        gender_vec = x[:, 16:20]

        projected = self.input_proj(x)
        gate      = self.gender_gate(gender_vec)
        gated     = projected * gate

        seq = gated.unsqueeze(1)
        out = self.transformer(seq)
        out = out[:, -1, :]
        return self.head(out)


# ---------------------------------------------------------------------------
# Training utils
# ---------------------------------------------------------------------------
class MeasurementLoss(nn.Module):
    """Weighted MSE — chest and waist weighted higher (harder to estimate)."""

    def __init__(self):
        super().__init__()
        # [chest, waist, hip, shoulder, sleeve, inseam]
        self.weights = torch.tensor(
            [1.5, 1.5, 1.2, 1.0, 1.0, 1.0], dtype=torch.float32
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.weights.to(pred.device)
        return (w * (pred - target) ** 2).mean()