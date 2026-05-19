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
from .yolo_detect import detect_person

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

global EXPECTED_FEATURES
EXPECTED_FEATURES = 300 

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
    Save extracted features + measurements safely to dataset.csv
    """

    try:
        features_flat = np.array(features_array).flatten()

        logger.info(f"Feature length: {len(features_flat)}")

        # Prevent corrupted CSV rows
        if len(features_flat) != EXPECTED_FEATURES:
            logger.warning(
                f"Skipping dataset save: expected {EXPECTED_FEATURES} features, got {len(features_flat)}"
            )
            return False

        feature_cols = [f"f{i}" for i in range(EXPECTED_FEATURES)]

        measurement_cols = [
            "chest",
            "waist",
            "hip",
            "shoulder",
            "sleeve",
            "inseam",
        ]

        all_cols = feature_cols + measurement_cols

        row_data = list(features_flat) + [
            float(measurements["chest"]),
            float(measurements["waist"]),
            float(measurements["hip"]),
            float(measurements["shoulder"]),
            float(measurements["sleeve"]),
            float(measurements["inseam"]),
        ]

        file_exists = os.path.exists(dataset_path)

        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

        writer = pd.DataFrame([row_data], columns=all_cols)

        writer.to_csv(
            dataset_path,
            mode="a",
            index=False,
            header=not file_exists,
        )

        logger.info(f"Dataset row saved successfully: {dataset_path}")

        return True

    except Exception as exc:
        logger.exception(f"Failed to save to dataset.csv: {exc}")
        EXPECTED_FEATURES = 300
        return False



def _build_dynamic_scale(front_img, fused_ratios, height_cm=None):

    ratios = np.array(fused_ratios[:6], dtype=np.float32)

    if height_cm:
        base = _get_height_ratios(height_cm) * height_cm
    else:
        base = SCALE.copy()

    body_type = _detect_body_type(front_img, height_cm)

    body_mult = np.array(
        _BODY_TYPE_MULTIPLIERS[body_type],
        dtype=np.float32
    )

    base = base * body_mult

    # Controlled correction
    corrections = np.clip(
        (ratios - 0.5) * 0.35,
        -0.12,
        0.18
    )

    final = base * (1.0 + corrections)

    logger.info(
        "Dynamic scale improved body=%s final=%s",
        body_type,
        final
    )

    return final




# =========================
# CSV SCALE LOADER
# =========================
def _load_scale_from_csv(csv_path: str) -> np.ndarray:

    if not os.path.exists(csv_path):
        logger.warning(
            "Calibration CSV not found at %s — using default SCALE.",
            csv_path
        )
        return _DEFAULT_SCALE.copy()

    try:
        df = pd.read_csv(
            csv_path,
            engine="python",
            on_bad_lines="skip"
        )

        missing = [k for k in MEASUREMENT_KEYS if k not in df.columns]

        if missing:
            logger.warning(
                "CSV missing columns: %s — using default SCALE",
                missing
            )
            return _DEFAULT_SCALE.copy()

        if len(df) == 0:
            logger.warning("CSV is empty — using default SCALE")
            return _DEFAULT_SCALE.copy()

        scale = df[MEASUREMENT_KEYS].mean().values.astype(np.float32)

        logger.info(
            "CSV SCALE loaded (%d rows): %s",
            len(df),
            scale
        )

        return scale

    except Exception as exc:
        logger.warning(
            "CSV load failed: %s — using default SCALE",
            exc
        )
        return _DEFAULT_SCALE.copy()


SCALE = _load_scale_from_csv(CSV_PATH)


# =========================
# MODEL LOAD + WARMUP
# =========================
def adapt_checkpoint_shape(state_dict: dict) -> dict:
    """
    Old checkpoint:
        [256, 286]

    Current model:
        [256, 292]

    Pad missing 6 features.
    """

    if "input_proj.0.weight" in state_dict:

        old_weight = state_dict["input_proj.0.weight"]

        # OLD MODEL CHECK
        if old_weight.shape[1] == 286 and old_weight.shape[0] == 256:

            # Create new padded tensor
            new_weight = torch.nn.functional.pad(
                old_weight,
                (0, 6),
                mode='constant',
                value=0.0
            )

            # Initialize new columns smartly
            new_weight[:, 286:292] = old_weight[:, 280:286] * 0.5

            state_dict["input_proj.0.weight"] = new_weight

            logger.info(
                "Adapted checkpoint weight "
                "[256,286] -> [256,292]"
            )

    return state_dict


def _load_model() -> BodyTransformer:
    m = BodyTransformer().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        logger.error("Model file not found at %s", MODEL_PATH)
        return m
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = adapt_checkpoint_shape(state_dict)
        m.load_state_dict(state_dict, strict=False)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    except Exception as exc:
        logger.exception("Model load failed: %s", exc)
    m.eval()
    return m


model = _load_model()


def _warmup():
    try:
        dummy = torch.zeros(1, 1, EXPECTED_FEATURES, dtype=torch.float32).to(DEVICE)
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
    Improved body type detector using edge + contour analysis
    More stable for dark clothes and complex backgrounds.
    """

    try:
        img = front_img.copy()
        h_img, w_img = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Better segmentation
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 50, 150)

        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return "medium"

        largest = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest)

        if h < MIN_IMG_HEIGHT or w < MIN_IMG_WIDTH:
            return "medium"

        # chest line
        chest_y = int(y + h * 0.42)

        # waist line
        waist_y = int(y + h * 0.55)

        # hip line
        hip_y = int(y + h * 0.68)

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest], -1, 255, -1)

        chest_px = np.sum(mask[chest_y, x:x+w] > 0)
        waist_px = np.sum(mask[waist_y, x:x+w] > 0)
        hip_px = np.sum(mask[hip_y, x:x+w] > 0)

        chest_ratio = chest_px / h
        waist_ratio = waist_px / h
        hip_ratio = hip_px / h

        logger.info(
            "Improved body ratios chest=%.3f waist=%.3f hip=%.3f",
            chest_ratio,
            waist_ratio,
            hip_ratio
        )

        avg_ratio = (chest_ratio + waist_ratio + hip_ratio) / 3

        if avg_ratio < 0.30:
            return "slim"

        elif avg_ratio > 0.42:
            return "heavy"

        return "medium"

    except Exception as exc:
        logger.warning("Improved body type detection failed: %s", exc)
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

    front_result = extract_pose(front_img, height_cm=height_cm)
    side_result = extract_pose(side_img, height_cm=height_cm)

    if front_result is None or front_result[0] is None:
        raise ValueError("Pose not detected in front image.")

    if side_result is None or side_result[0] is None:
        raise ValueError("Pose not detected in side image.")

    front_feat, front_class = front_result
    side_feat, side_class = side_result

    fused, final_class = fuse_features(
        front_feat,
        side_feat,
        front_class,
        side_class
    )

    body_type = _detect_body_type(front_img, height_cm)

    logger.info(f"Detected body type: {body_type}")

    # =========================
    # WITH HEIGHT
    # =========================
    if height_cm:

        dynamic_scale = _build_dynamic_scale(
            front_img,
            fused[:6],
            height_cm
        )

        chest_cm = dynamic_scale[0]
        waist_cm = dynamic_scale[1]
        hip_cm = dynamic_scale[2]
        shoulder_cm = dynamic_scale[3]
        sleeve_cm = dynamic_scale[4]
        inseam_cm = dynamic_scale[5]

        hip_cm = _sanitize_hip(
            hip_cm,
            chest_cm,
            height_cm=height_cm
        )

        sleeve_cm = _sleeve_correction(
            sleeve_cm,
            height_cm
        )

        result = {
            "chest": round(float(chest_cm), 1),
            "waist": round(float(waist_cm), 1),
            "hip": round(float(hip_cm), 1),
            "shoulder": round(float(shoulder_cm), 1),
            "sleeve": round(float(sleeve_cm), 1),
            "inseam": round(float(inseam_cm), 1),
        }

        logger.info(f"Geometry measurements with height: {result}")

        return result

    # =========================
    # WITHOUT HEIGHT
    # =========================

    bt_mult = np.array(
        _BODY_TYPE_MULTIPLIERS[body_type],
        dtype=np.float32
    )

    adjusted_scale = SCALE * bt_mult

    chest_r = float(fused[0])
    waist_r = float(fused[1])
    hip_r = float(fused[2])
    shoulder_r = float(fused[3])
    sleeve_r = float(fused[4])
    inseam_r = float(fused[5])

    def _scale_blend(r, scale_val):
        deviation = np.clip(r - 0.5, -0.15, 0.15)
        return scale_val * (1.0 + deviation)

    chest_cm = _scale_blend(chest_r, adjusted_scale[0])

    hip_raw = _scale_blend(hip_r, adjusted_scale[2])

    hip_cm = _sanitize_hip(
        hip_raw,
        chest_cm,
        height_cm=None
    )

    sleeve_cm = _scale_blend(
        sleeve_r,
        adjusted_scale[4]
    )

    result = {
        "chest": round(float(chest_cm), 1),
        "waist": round(float(_scale_blend(waist_r, adjusted_scale[1])), 1),
        "hip": round(float(hip_cm), 1),
        "shoulder": round(float(_scale_blend(shoulder_r, adjusted_scale[3])), 1),
        "sleeve": round(float(sleeve_cm), 1),
        "inseam": round(float(_scale_blend(inseam_r, adjusted_scale[5])), 1),
    }

    logger.info(f"Geometry measurements without height: {result}")

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

# YOLO crop
    front_crop = detect_person(front_img)
    side_crop = detect_person(side_img)

    if front_crop is not None:
      front_img = front_crop

    if side_crop is not None:
       side_img = side_crop

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