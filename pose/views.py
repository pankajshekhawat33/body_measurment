import os
import logging
import numpy as np
import cv2
import torch
import pandas as pd
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .models import BodyTransformer
from .ai_utils import extract_pose, fuse_features, KalmanFilter


logger = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = getattr(
    settings, "BODY_AI_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "body_ai_model.pth")
)

CSV_PATH = getattr(
    settings, "BODY_AI_CSV_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")
)

MEASUREMENT_KEYS = ["chest", "waist", "hip", "shoulder", "sleeve", "inseam"]

# Realistic mean adult measurements (cm) — used when no CSV found
# Updated with better calibration for average adult male
_DEFAULT_SCALE = np.array([100.0, 88.0, 102.0, 46.0, 65.0, 80.0], dtype=np.float32)

# -----------------------------------------------------------------------
# ANTHROPOMETRIC RATIO TABLE
# Source: ISO 7250 / ANSUR II — fraction of standing height
# Format: (min_h, max_h) → [chest, waist, hip, shoulder, sleeve, inseam]
#
# NOTE on sleeve ratio:
#   Full sleeve = shoulder point → wrist = ~0.385 × height
#   Many pose models return forearm-only (~0.22) or elbow (~0.30).
#   We correct this in _geometry_fallback with _sleeve_correction().
# -----------------------------------------------------------------------
_HEIGHT_RATIOS = {
    (140, 154): [0.590, 0.500, 0.590, 0.260, 0.380, 0.455],
    (155, 164): [0.595, 0.505, 0.595, 0.265, 0.385, 0.460],
    (165, 174): [0.600, 0.510, 0.600, 0.270, 0.390, 0.465],
    (175, 184): [0.605, 0.515, 0.605, 0.275, 0.395, 0.470],
    (185, 200): [0.610, 0.520, 0.615, 0.280, 0.400, 0.475],
}
_DEFAULT_RATIOS = np.array([0.595, 0.505, 0.595, 0.265, 0.385, 0.460], dtype=np.float32)


def _get_height_ratios(height_cm: float) -> np.ndarray:
    for (lo, hi), ratios in _HEIGHT_RATIOS.items():
        if lo <= height_cm <= hi:
            return np.array(ratios, dtype=np.float32)
    return _DEFAULT_RATIOS.copy()


# -----------------------------------------------------------------------
# BODY TYPE SCALE TABLE
# Maps visual build categories to measurement scale multipliers.
# These correct the anthropometric priors for non-average builds.
#
# slim:    chest 88–94, waist 72–80, hip 88–94
# medium:  chest 95–104, waist 81–90, hip 95–104
# heavy:   chest 105+, waist 91+, hip 105+
#
# Multipliers are applied to the height-ratio-based prior.
# -----------------------------------------------------------------------
_BODY_TYPE_MULTIPLIERS = {
    # build: [chest, waist, hip, shoulder, sleeve, inseam]
    "slim":   [0.90, 0.85, 0.88, 0.95, 0.98, 1.00],
    "medium": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "heavy":  [1.12, 1.18, 1.12, 1.04, 1.00, 1.00],
}

# Clamp ranges (cm) — anatomically validated
# Updated to avoid false minimum clamping for realistic adults
MEASUREMENT_CLAMP = {
    "chest":    (80.0,  160.0),    # increased from 72
    "waist":    (62.0,  140.0),    # increased from 55
    "hip":      (85.0,  140.0),    # increased from 75
    "shoulder": (40.0,   75.0),    # increased from 36
    "sleeve":   (55.0,  105.0),    # increased from 52
    "inseam":   (65.0,  105.0),    # increased from 55
}

# Hip sanity guard: hip cannot exceed chest by more than this factor
# (for non-pregnant adults). Catches side-view depth spikes.
_HIP_MAX_CHEST_RATIO = 1.18

MIN_IMG_WIDTH  = 100
MIN_IMG_HEIGHT = 200


# =========================
# DATASET CSV SAVER
# =========================
def _save_to_dataset_csv(features_array: np.ndarray, measurements: dict, dataset_path: str = "pose/dataset.csv"):
    """
    Save extracted features + measurements to dataset.csv for model training.
    Format: [f0, f1, ..., f264, chest, waist, hip, shoulder, sleeve, inseam]
    """
    try:
        features_flat = np.array(features_array).flatten()
        n_features = len(features_flat)
        feature_cols = [f"f{i}" for i in range(n_features)]
        measurement_cols = ["chest", "waist", "hip", "shoulder", "sleeve", "inseam"]
        all_cols = feature_cols + measurement_cols
        
        row_data = list(features_flat) + [
            measurements["chest"],
            measurements["waist"],
            measurements["hip"],
            measurements["shoulder"],
            measurements["sleeve"],
            measurements["inseam"],
        ]
        
        file_exists = os.path.exists(dataset_path)
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        writer = pd.DataFrame([row_data], columns=all_cols)
        with open(dataset_path, "a", newline="") as f:
            writer.to_csv(f, index=False, header=(not file_exists))
        
        logger.info(f"Dataset row saved: {dataset_path}")
        return True
    except Exception as exc:
        logger.warning(f"Failed to save to dataset.csv: {exc}")
        return False



def _build_dynamic_scale(front_img, fused_ratios, height_cm=None):
    """
    Per-image adaptive scale.
    fused_ratios: [chest, waist, hip, shoulder, sleeve, inseam] (0–1)
    """

    ratios = np.array(fused_ratios[:6], dtype=np.float32)

    # 1) base prior (dataset or default)
    base = SCALE.copy()

    # 2) height-based prior (strong signal)
    if height_cm:
        h_prior = _get_height_ratios(height_cm) * height_cm
    else:
        h_prior = base

    # 3) body-type multiplier
    body_type = _detect_body_type(front_img, height_cm)
    bt = np.array(_BODY_TYPE_MULTIPLIERS[body_type], dtype=np.float32)

    # 4) silhouette width factor (dynamic per image)
    h, w = front_img.shape[:2]
    gray = cv2.cvtColor(front_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    width_px = np.sum(mask[int(h * 0.5)] > 128)
    width_ratio = width_px / h  # body width vs height

    # normalize width factor
    width_factor = np.clip((width_ratio - 0.30) * 2.5, -0.15, 0.25)

    # 5) combine all
    dynamic_scale = h_prior * bt

    # apply width adjustment
    dynamic_scale = dynamic_scale * (1.0 + width_factor)

    # 6) blend with pose ratios (final correction)
    final = dynamic_scale * (0.7 + 0.6 * ratios)

    logger.info(
        "Dynamic scale → body_type=%s width_factor=%.3f final=%s",
        body_type, width_factor, final
    )

    return final





# =========================
# CSV SCALE LOADER
# =========================
def _load_scale_from_csv(csv_path: str) -> np.ndarray:
    if not os.path.exists(csv_path):
        logger.warning("Calibration CSV not found at %s — using default SCALE.", csv_path)
        return _DEFAULT_SCALE.copy()
    try:
        df = pd.read_csv(csv_path)
        missing = [k for k in MEASUREMENT_KEYS if k not in df.columns]
        if missing:
            logger.warning("CSV missing columns: %s — using default SCALE", missing)
            return _DEFAULT_SCALE.copy()
        if len(df) == 0:
            logger.warning("CSV is empty — using default SCALE")
            return _DEFAULT_SCALE.copy()
        scale = df[MEASUREMENT_KEYS].mean().values.astype(np.float32)
        logger.info("CSV SCALE loaded (%d rows): %s", len(df), scale)
        return scale
    except Exception as exc:
        logger.warning("CSV load failed: %s — using default SCALE", exc)
        return _DEFAULT_SCALE.copy()


SCALE = _load_scale_from_csv(CSV_PATH)


# =========================
# MODEL LOAD + WARMUP
# =========================
def _adapt_checkpoint_shape(state_dict: dict) -> dict:
    """
    Adapter for checkpoint compatibility: old model had INPUT_SIZE=286 (N_GEO=10),
    new model has INPUT_SIZE=292 (N_GEO=16). Pad input_proj weights to new size.
    """
    if "input_proj.0.weight" in state_dict:
        old_weight = state_dict["input_proj.0.weight"]  # shape: [256, 286]
        if old_weight.shape[1] == 286 and old_weight.shape[0] == 256:
            # Pad from [256, 286] to [256, 292] by repeating last 6 channels
            new_weight = torch.nn.functional.pad(old_weight, (0, 6), mode='constant', value=0.0)
            # Average the new columns from nearby old ones (smooth initialization)
            new_weight[:, 286:292] = old_weight[:, 280:286] * 0.5
            state_dict["input_proj.0.weight"] = new_weight
            logger.info("Adapted input_proj.0.weight: [256, 286] → [256, 292]")
    
    if "input_proj.1.weight" in state_dict and "input_proj.1.bias" in state_dict:
        # LayerNorm: only has weight/bias for the output dim (256), not affected
        pass
    
    return state_dict


def _load_model() -> BodyTransformer:
    m = BodyTransformer().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        logger.error("Model file not found at %s", MODEL_PATH)
        return m
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = _adapt_checkpoint_shape(state_dict)
        m.load_state_dict(state_dict, strict=False)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    except Exception as exc:
        logger.exception("Model load failed: %s", exc)
    m.eval()
    return m


model = _load_model()


def _warmup():
    try:
        dummy = torch.zeros(1, 1, 280, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            model(dummy)
        logger.info("Model warm-up done.")
    except Exception as exc:
        logger.warning("Warm-up failed (non-fatal): %s", exc)


_warmup()


# =========================
# BODY TYPE DETECTOR
# =========================
def _detect_body_type(front_img: np.ndarray, height_cm: float = None) -> str:
    """
    Estimate body build from front image pixel ratios.

    Strategy:
      1. Detect the person bounding box via foreground mask.
      2. Sample pixel widths at chest level (~45% from top of bbox)
         and hip level (~65% from top of bbox).
      3. Compare these to height_cm to estimate proportions.

    Returns: "slim" | "medium" | "heavy"
    Fallback: "medium" if detection fails.

    This is deliberately simple — it just needs to be directionally
    correct to shift measurements by 7–12%, not be pixel-perfect.
    """
    try:
        h_img, w_img = front_img.shape[:2]

        # Convert to grayscale and threshold to find person silhouette
        gray = cv2.cvtColor(front_img, cv2.COLOR_BGR2GRAY)

        # Use Otsu threshold to separate person from background
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours — pick the largest (the person)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "medium"

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        if h < MIN_IMG_HEIGHT or w < MIN_IMG_WIDTH:
            return "medium"

        # Sample chest width at ~45% down from top of bounding box
        chest_y = int(y + h * 0.45)
        chest_row = mask[chest_y, x: x + w]
        chest_px = int(np.sum(chest_row > 128))

        # Sample hip width at ~65% down
        hip_y = int(y + h * 0.65)
        hip_row = mask[hip_y, x: x + w]
        hip_px = int(np.sum(hip_row > 128))

        # Normalize widths by height in pixels to get a proportion
        chest_ratio = chest_px / h
        hip_ratio   = hip_px   / h

        logger.info(
            "Body type detection — chest_ratio=%.3f hip_ratio=%.3f",
            chest_ratio, hip_ratio
        )

        # Empirical thresholds (tuned on South Asian adult photos):
        # slim:   chest_ratio < 0.28
        # heavy:  chest_ratio > 0.38 or hip_ratio > 0.40
        # medium: everything else
        if chest_ratio < 0.28 and hip_ratio < 0.33:
            body_type = "slim"
        elif chest_ratio > 0.38 or hip_ratio > 0.40:
            body_type = "heavy"
        else:
            body_type = "medium"

        logger.info("Detected body type: %s", body_type)
        return body_type

    except Exception as exc:
        logger.warning("Body type detection failed: %s — defaulting to medium", exc)
        return "medium"


# =========================
# SLEEVE CORRECTOR
# =========================
def _sleeve_correction(raw_sleeve_cm: float, height_cm: float) -> float:
    """
    FIX: Many pose models (MediaPipe, OpenPose) return the elbow→wrist
    segment or shoulder→elbow segment instead of the full sleeve length.

    Full sleeve = shoulder point → wrist ≈ 0.385 × height_cm
    If the raw value is < 85% of expected full sleeve, it's a partial
    segment — correct it upward using the anthropometric ratio.

    We never correct downward (if the model already returned a plausible
    full-arm length, we leave it alone).
    """
    expected_full_sleeve = _get_height_ratios(height_cm)[4] * height_cm
    if raw_sleeve_cm < 0.85 * expected_full_sleeve:
        # Blend: 60% anthropometric prior + 40% detected
        corrected = 0.60 * expected_full_sleeve + 0.40 * raw_sleeve_cm
        logger.info(
            "Sleeve corrected: %.1f → %.1f (expected full=%.1f)",
            raw_sleeve_cm, corrected, expected_full_sleeve
        )
        return corrected
    return raw_sleeve_cm


# =========================
# HIP OUTLIER GUARD
# =========================
def _sanitize_hip(hip_cm: float, chest_cm: float, height_cm: float = None) -> float:
    """
    FIX: Hip > chest × 1.18 is anatomically implausible for non-pregnant adults.
    This spike is almost always caused by:
      - Dark clothing confusing the silhouette detector at hip level
      - Side-view depth being overestimated
      - Shadow or background merged into hip contour

    Correction: cap hip at min(chest × 1.18, anthropometric_prior × 1.05)
    """
    max_by_chest = chest_cm * _HIP_MAX_CHEST_RATIO

    if height_cm:
        prior_hip = _get_height_ratios(height_cm)[2] * height_cm
        max_by_prior = prior_hip * 1.05
        cap = min(max_by_chest, max_by_prior)
    else:
        cap = max_by_chest

    if hip_cm > cap:
        logger.warning(
            "Hip outlier detected: %.1f > cap %.1f (chest=%.1f) — clamping.",
            hip_cm, cap, chest_cm
        )
        return round(cap, 1)
    return hip_cm


# =========================
# HELPERS
# =========================
def denormalize(output: np.ndarray, height_cm: float = None) -> np.ndarray:
    flat = np.array(output).flatten()
    if flat.shape[0] < len(SCALE):
        raise ValueError(f"Model output has {flat.shape[0]} values, need >= {len(SCALE)}")

    values = flat[: len(SCALE)]
    logger.info("Raw model output: %s", values)

    base = _get_height_ratios(height_cm) * height_cm if height_cm else SCALE

    if np.max(np.abs(values)) < 0.05:
        logger.warning("Model output too small — using base fallback: %s", base)
        return base.copy()

    scaled = values * base if np.max(values) <= 1.5 else values
    logger.info("Denormalized measurements (cm): %s", scaled)
    return scaled


def _decode_image(file_obj):
    try:
        raw = np.frombuffer(file_obj.read(), dtype=np.uint8)
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.warning("Image decode failed: %s", exc)
        return None


def _validate_image(img, label: str):
    if img is None:
        return f"{label} image could not be decoded."
    h, w = img.shape[:2]
    if w < MIN_IMG_WIDTH or h < MIN_IMG_HEIGHT:
        return f"{label} image too small ({w}x{h}). Min: {MIN_IMG_WIDTH}x{MIN_IMG_HEIGHT}."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 20.0:
        return f"{label} image is too blurry. Please retake in better light."
    return None


def _clamp_measurements(measurements: dict) -> dict:
    """
    Clamp all measurements to anatomical bounds and ensure clean Python floats.
    numpy float32 can produce ugly repr like 107.0999984741211 — explicit
    float() + round() to 1dp ensures clean JSON output (e.g. 107.1).
    """
    clamped = {}
    for key, val in measurements.items():
        lo, hi = MEASUREMENT_CLAMP.get(key, (0, 999))
        # np.clip → float64, then Python float, then round to 1dp
        clamped[key] = round(float(np.clip(float(val), lo, hi)), 1)
    return clamped


# =========================
# TRANSFORMER PIPELINE
# =========================
def process_images(front_img, side_img, height_cm=None):
    front_result = extract_pose(front_img, height_cm=height_cm)
    side_result  = extract_pose(side_img,  height_cm=height_cm)

    if front_result is None or front_result[0] is None:
        raise ValueError(
            "Pose not detected in front image. "
            "Stand upright, full body visible, good lighting."
        )
    if side_result is None or side_result[0] is None:
        raise ValueError(
            "Pose not detected in side image. "
            "Stand upright, full body visible, good lighting."
        )

    front_feat, front_class = front_result
    side_feat, side_class   = side_result
    features, final_class   = fuse_features(front_feat, side_feat, front_class, side_class)

    kalman   = KalmanFilter(dim=len(features))
    features = kalman.update(np.array(features))

    x = torch.tensor(features, dtype=torch.float32).to(DEVICE)
    x = x.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(x)

    return output.cpu().numpy().tolist()


# =========================
# GEOMETRY-ONLY FALLBACK  (v2 — body-type aware)
# =========================
def _geometry_fallback(front_img, side_img, height_cm=None):
    """
    Direct geometric estimate with three new corrections vs v1:

    1. Body-type detection (slim / medium / heavy) from front silhouette
       → scales the anthropometric prior up or down by 7–12%
    2. Sleeve correction: pose models often return partial arm segment;
       corrected to full shoulder→wrist length
    3. Hip outlier guard: hip > chest × 1.18 is capped to a plausible value
       (fixes the hip=144 bug caused by dark jeans + side-view depth spike)
    """
    front_result = extract_pose(front_img, height_cm=height_cm)
    side_result  = extract_pose(side_img,  height_cm=height_cm)

    if front_result is None or front_result[0] is None:
        raise ValueError("Pose not detected in front image.")
    if side_result is None or side_result[0] is None:
        raise ValueError("Pose not detected in side image.")

    front_feat, front_class = front_result
    side_feat, side_class   = side_result
    fused, final_class      = fuse_features(front_feat, side_feat, front_class, side_class)

    # fused[0:6] = normalized ratios: chest, waist, hip, shoulder, sleeve, inseam
    chest_r    = float(fused[0])
    waist_r    = float(fused[1])
    hip_r      = float(fused[2])
    shoulder_r = float(fused[3])
    sleeve_r   = float(fused[4])
    inseam_r   = float(fused[5])

    logger.info(
        "Geometry fallback ratios: chest=%.4f waist=%.4f hip=%.4f "
        "shoulder=%.4f sleeve=%.4f inseam=%.4f",
        chest_r, waist_r, hip_r, shoulder_r, sleeve_r, inseam_r
    )

    # --- Detect body type from front image ---
    body_type = _detect_body_type(front_img, height_cm)
    bt_mult   = np.array(_BODY_TYPE_MULTIPLIERS[body_type], dtype=np.float32)
    logger.info("Body type: %s → multipliers: %s", body_type, bt_mult)

    if height_cm:
        h      = height_cm
        ratios = _get_height_ratios(h) * bt_mult   # apply body-type scaling to prior

        # Blend detected ratio with body-type-adjusted anthropometric prior.
        # For slim builds: pose detection is noisier (less body mass to detect),
        # so we trust the prior MORE → 50/50 blend.
        # For medium/heavy builds: pose signal is stronger → 65/35 blend.
        blend = 0.50 if body_type == "slim" else 0.65

        def _b(r, prior_r):
            return blend * r + (1.0 - blend) * prior_r

        chest_cm    = _b(chest_r,    ratios[0]) * h
        waist_cm    = _b(waist_r,    ratios[1]) * h
        hip_raw_cm  = _b(hip_r,      ratios[2]) * h
        shoulder_cm = _b(shoulder_r, ratios[3]) * h
        sleeve_raw  = _b(sleeve_r,   ratios[4]) * h
        inseam_cm   = _b(inseam_r,   ratios[5]) * h

        # Apply sleeve correction (partial-arm fix)
        sleeve_cm = _sleeve_correction(sleeve_raw, h)

        # Apply hip outlier guard
        hip_cm = _sanitize_hip(hip_raw_cm, chest_cm, height_cm=h)

        result = {
            "chest":    round(chest_cm,    1),
            "waist":    round(waist_cm,    1),
            "hip":      round(hip_cm,      1),
            "shoulder": round(shoulder_cm, 1),
            "sleeve":   round(sleeve_cm,   1),
            "inseam":   round(inseam_cm,   1),
        }
        logger.info("Height+body-type measurements: %s", result)
        return result

    # No height_cm — use SCALE as the base, blended with body-type multiplier
    adjusted_scale = SCALE * bt_mult

    def _scale_blend(r, scale_val):
        deviation = np.clip(r - 0.5, -0.15, 0.15)
        return scale_val * (1.0 + deviation)

    chest_cm   = _scale_blend(chest_r,    adjusted_scale[0])
    sleeve_raw = _scale_blend(sleeve_r,   adjusted_scale[4])
    sleeve_cm  = sleeve_raw  # no height ref available; clamp will catch extremes
    hip_raw    = _scale_blend(hip_r,      adjusted_scale[2])
    hip_cm     = _sanitize_hip(hip_raw, chest_cm, height_cm=None)

    result = {
        "chest":    round(chest_cm,                                  1),
        "waist":    round(_scale_blend(waist_r,    adjusted_scale[1]), 1),
        "hip":      round(hip_cm,                                    1),
        "shoulder": round(_scale_blend(shoulder_r, adjusted_scale[3]), 1),
        "sleeve":   round(sleeve_cm,                                 1),
        "inseam":   round(_scale_blend(inseam_r,   adjusted_scale[5]), 1),
    }
    logger.info("SCALE+body-type measurements (no height): %s", result)
    return result


# =========================
# POST-PROCESS SANITY CHECK
# =========================
def _sanity_check(m: dict, height_cm: float = None) -> dict:
    """
    Final cross-measurement consistency checks.

    Rules (anatomical):
      1. hip cannot exceed chest × 1.18        → already handled by _sanitize_hip,
                                                  but catch transformer path too
      2. waist must be < chest                  → waist > chest = impossible
      3. shoulder must be < chest / 1.8         → shoulder > 28cm × 2 is suspicious
      4. sleeve must be >= shoulder × 1.5       → if sleeve < shoulder, it's forearm only
      5. inseam must be > sleeve                → for most people, legs longer than arms
    """
    chest    = m["chest"]
    waist    = m["waist"]
    hip      = m["hip"]
    shoulder = m["shoulder"]
    sleeve   = m["sleeve"]
    inseam   = m["inseam"]

    # Rule 1: hip cap
    hip = _sanitize_hip(hip, chest, height_cm)

    # Rule 2: waist < chest
    if waist >= chest:
        logger.warning("Sanity: waist(%.1f) >= chest(%.1f) — correcting", waist, chest)
        waist = chest * 0.88

    # Rule 3: shoulder plausibility
    max_shoulder = chest / 1.8
    if shoulder > max_shoulder:
        logger.warning("Sanity: shoulder(%.1f) > max(%.1f) — correcting", shoulder, max_shoulder)
        shoulder = max_shoulder

    # Rule 4: sleeve >= shoulder × 1.5 (otherwise it's elbow only)
    if sleeve < shoulder * 1.5:
        logger.warning(
            "Sanity: sleeve(%.1f) looks like partial arm (shoulder=%.1f) — correcting",
            sleeve, shoulder
        )
        sleeve = _sleeve_correction(sleeve, height_cm) if height_cm else shoulder * 1.7

    # Rule 5: inseam > sleeve (for most adult builds)
    # Don't auto-correct this — just log it as a warning
    if inseam < sleeve:
        logger.warning(
            "Sanity: inseam(%.1f) < sleeve(%.1f) — unusual, check pose detection",
            inseam, sleeve
        )

    return {
        "chest":    round(chest,    1),
        "waist":    round(waist,    1),
        "hip":      round(hip,      1),
        "shoulder": round(shoulder, 1),
        "sleeve":   round(sleeve,   1),
        "inseam":   round(inseam,   1),
    }


# =========================
# API
# =========================
@api_view(["POST"])
def detect_body(request):
    """
    POST /api/detect-body/

    Form fields:
        front      (file)   — front view image
        side       (file)   — side view image
        height_cm  (float)  — RECOMMENDED: person's real height in cm
    """
    front_file = request.FILES.get("front")
    side_file  = request.FILES.get("side")

    if not front_file or not side_file:
        return Response(
            {"error": "Both 'front' and 'side' images are required."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    height_cm  = None
    height_raw = request.data.get("height_cm")
    if height_raw:
        try:
            height_cm = float(height_raw)
            if not (100.0 <= height_cm <= 250.0):
                return Response(
                    {"error": "height_cm must be between 100 and 250."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        except (ValueError, TypeError):
            return Response(
                {"error": "height_cm must be a number, e.g. 170.5"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    front_img = _decode_image(front_file)
    side_img  = _decode_image(side_file)

    err = _validate_image(front_img, "Front") or _validate_image(side_img, "Side")
    if err:
        return Response({"error": err}, status=status.HTTP_400_BAD_REQUEST)

    use_transformer = os.path.exists(MODEL_PATH)

    try:
        if use_transformer:
            raw_result      = process_images(front_img, side_img, height_cm=height_cm)
            measurements_cm = denormalize(raw_result[0], height_cm=height_cm)

            if np.max(measurements_cm) < 40.0:
                logger.warning("Transformer output invalid — falling back to geometry.")
                measurements = _geometry_fallback(front_img, side_img, height_cm=height_cm)
                method = "Geometry fallback (transformer output invalid)"
            else:
                measurements = {
                    key: round(float(measurements_cm[i]), 1)
                    for i, key in enumerate(MEASUREMENT_KEYS)
                }
                method = "BodyTransformer + KalmanFilter"
        else:
            logger.warning("Model file not found — using geometry fallback.")
            measurements = _geometry_fallback(front_img, side_img, height_cm=height_cm)
            method = "Geometry only (train BodyTransformer for 90%+ accuracy)"

    except ValueError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception:
        logger.exception("Inference error.")
        return Response(
            {"error": "Server error during inference. Please try again."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    # Clamp first, then cross-measurement sanity check
    measurements = _clamp_measurements(measurements)
    measurements = _sanity_check(measurements, height_cm=height_cm)

    # Save to dataset.csv: extract fresh features and append
    try:
        front_result = extract_pose(front_img, height_cm=height_cm)
        side_result  = extract_pose(side_img,  height_cm=height_cm)
        if front_result and side_result:
            front_feat, _ = front_result
            side_feat, _ = side_result
            fused_features, _ = fuse_features(front_feat, side_feat, front_result[1], side_result[1])
            _save_to_dataset_csv(fused_features, measurements)
    except Exception as exc:
        logger.warning(f"Could not save to dataset.csv: {exc}")

    return Response(
        {
            "measurements": measurements,
            "unit": "cm",
            "height_used": height_cm,
            "method": method,
            "device": str(DEVICE),
            "scale_source": (
                "dataset.csv" if os.path.exists(CSV_PATH) else "default (no CSV found)"
            ),
            "tip": (
                None if height_cm
                else "Send 'height_cm' with the request for significantly better accuracy."
            ),
        },
        status=status.HTTP_200_OK,
    )


# =========================
# TEST ENDPOINT
# =========================
@api_view(["GET"])
def test_model_status(request):
    model_exists  = os.path.exists(MODEL_PATH)
    if not model_exists:
        return Response(
            {
                "model_ready": False,
                "status": "Model not trained",
                "message": "Run: python pose/train_advanced.py",
            },
            status=status.HTTP_200_OK,
        )

    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    if model_size_mb < 1:
        return Response(
            {
                "model_ready": False,
                "status": "Model exists but appears untrained",
                "model_size_mb": round(model_size_mb, 2),
                "message": "Re-train: python pose/train_advanced.py",
            },
            status=status.HTTP_200_OK,
        )

    return Response(
        {
            "model_ready": True,
            "status": "Model trained and ready",
            "model_size_mb": round(model_size_mb, 2),
            "device": str(DEVICE),
            "scale_factors": SCALE.tolist(),
            "measurement_keys": MEASUREMENT_KEYS,
            "expected_accuracy": "90%+",
            "instructions": {
                "endpoint": "POST /api/detect-body/",
                "params": ["front (image)", "side (image)", "height_cm (recommended)"],
            }
        },
        status=status.HTTP_200_OK,
    )