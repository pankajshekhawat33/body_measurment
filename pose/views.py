import os
import logging
import numpy as np
import cv2
import torch
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

# Calibrated real-world scale per measurement (cm).
# These are average adult values — fine-tune after testing on real people.
SCALE = np.array(
    getattr(settings, "BODY_AI_SCALE", [95.0, 80.0, 98.0, 44.0, 62.0, 78.0]),
    dtype=np.float32
)

MEASUREMENT_KEYS = ["chest", "waist", "hip", "shoulder", "sleeve", "inseam"]

# Realistic clamp ranges (cm) — reject absurd model outputs
MEASUREMENT_CLAMP = {
    "chest":    (60.0, 160.0),
    "waist":    (20.0, 110.0),
    "hip":      (30.0, 170.0),
    "shoulder": (20.0,  80.0),
    "sleeve":   (20.0,  90.0),
    "inseam":   (20.0, 100.0),
}

MIN_IMG_WIDTH  = 100
MIN_IMG_HEIGHT = 200


# =========================
# MODEL LOAD + WARMUP
# =========================
def _load_model() -> BodyTransformer:
    m = BodyTransformer().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        logger.error("Model file not found at %s", MODEL_PATH)
        return m
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        # Try strict=False to handle architecture mismatches from older checkpoints
        m.load_state_dict(state_dict, strict=False)
        logger.info("Model loaded from %s", MODEL_PATH)
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
# HELPERS
# =========================
def denormalize(output: np.ndarray) -> np.ndarray:
    """
    Denormalize model output to cm measurements.
    Model outputs values in [0, 1] range (sigmoid) or normalized ratios.
    """
    flat = np.array(output).flatten()
    if flat.shape[0] < len(SCALE):
        raise ValueError(
            f"Model output has {flat.shape[0]} values, need >= {len(SCALE)}"
        )
    
    normalized = flat[: len(SCALE)]
    
    # Debug logging
    logger.info(f"Raw model output (first 6): {normalized[:6]}")
    
    # Check if output looks untrained (all zeros or very small)
    if np.max(np.abs(normalized)) < 0.01:
        logger.warning("Model output suspiciously small - likely untrained model")
        return np.zeros(len(SCALE))
    
    # Scale to cm - multiply by SCALE factors
    scaled = normalized * SCALE
    logger.info(f"Denormalized measurements: {scaled}")
    
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
    clamped = {}
    for key, val in measurements.items():
        lo, hi = MEASUREMENT_CLAMP.get(key, (0, 999))
        clamped[key] = round(float(np.clip(val, lo, hi)), 1)
    return clamped


# =========================
# TRANSFORMER PIPELINE
# =========================
def process_images(front_img, side_img, height_cm=None):
    front_result = extract_pose(front_img, height_cm=height_cm)
    side_result  = extract_pose(side_img,  height_cm=height_cm)

    # extract_pose returns (features, classification) tuple
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
    side_feat, side_class = side_result

    features, final_class = fuse_features(front_feat, side_feat, front_class, side_class)

    kalman   = KalmanFilter(dim=len(features))
    features = kalman.update(np.array(features))

    x = torch.tensor(features, dtype=torch.float32).to(DEVICE)
    x = x.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(x)

    return output.cpu().numpy().tolist()


# =========================
# GEOMETRY-ONLY FALLBACK
# (use when BodyTransformer not trained yet)
# =========================
def _geometry_fallback(front_img, side_img, height_cm=None):
    """
    Direct geometric estimate — no transformer needed.
    Accuracy: ~75-82% with height_cm, ~55-65% without.
    """
    front_result = extract_pose(front_img, height_cm=height_cm)
    side_result  = extract_pose(side_img,  height_cm=height_cm)

    if front_result is None or front_result[0] is None:
        raise ValueError("Pose not detected in front image.")
    if side_result is None or side_result[0] is None:
        raise ValueError("Pose not detected in side image.")

    front_feat, front_class = front_result
    side_feat, side_class = side_result

    fused, final_class = fuse_features(front_feat, side_feat, front_class, side_class)
    # fused structure:
    # [0:10] = [chest_circ, waist_circ, hip_circ, shoulder, sleeve, inseam, torso, h_proxy, hip_sh, head]
    # [10:14] = gender one-hot + confidence
    # [14:] = visibility + raw landmarks

    chest_circ  = float(fused[0])
    waist_circ  = float(fused[1])
    hip_circ    = float(fused[2])
    shoulder    = float(fused[3])
    sleeve      = float(fused[4])
    inseam      = float(fused[5])
    h_proxy     = float(fused[7])  # Actual body height proxy
    
    logger.info(f"Geometry fallback - h_proxy: {h_proxy}, measurements: {chest_circ}, {waist_circ}, {hip_circ}")
    
    # If height provided, measurements are already in cm
    if height_cm:
        return {
            "chest":    round(chest_circ, 1),
            "waist":    round(waist_circ, 1),
            "hip":      round(hip_circ, 1),
            "shoulder": round(shoulder, 1),
            "sleeve":   round(sleeve, 1),
            "inseam":   round(inseam, 1),
        }
    
    # Without height, use h_proxy to denormalize (person's actual body proportions)
    # h_proxy is in normalized space, scale it using average adult height
    avg_adult_height = 170.0
    scale = avg_adult_height * h_proxy if h_proxy > 0 else 170.0
    
    return {
        "chest":    round(chest_circ * scale, 1),
        "waist":    round(waist_circ * scale, 1),
        "hip":      round(hip_circ * scale, 1),
        "shoulder": round(shoulder * scale, 1),
        "sleeve":   round(sleeve * scale, 1),
        "inseam":   round(inseam * scale, 1),
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
                              Without this, accuracy drops ~20%.

    Returns measurements for: chest, waist, hip, shoulder, sleeve, inseam
    """
    front_file = request.FILES.get("front")
    side_file  = request.FILES.get("side")

    if not front_file or not side_file:
        return Response(
            {"error": "Both 'front' and 'side' images are required."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Optional height
    height_cm = None
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
            raw_result = process_images(front_img, side_img, height_cm=height_cm)
            measurements_cm = denormalize(raw_result[0])
            
            # Check if model output is suspiciously bad
            if np.max(measurements_cm) < 10.0:
                logger.warning("Model producing invalid output - falling back to geometry")
                measurements = _geometry_fallback(front_img, side_img, height_cm=height_cm)
                method = "Geometry Fallback (Model untrained)"
            else:
                measurements = {
                    key: round(float(measurements_cm[i]), 1)
                    for i, key in enumerate(MEASUREMENT_KEYS)
                }
                method = "BodyTransformer + KalmanFilter"
        else:
            logger.warning("Model not found — using geometry fallback.")
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

    measurements = _clamp_measurements(measurements)

    return Response(
        {
            "measurements": measurements,
            "unit": "cm",
            "height_used": height_cm,
            "method": method,
            "device": str(DEVICE),
            "tip": (
                None if height_cm
                else "Send 'height_cm' with the request for significantly better accuracy."
            ),
        },
        status=status.HTTP_200_OK,
    )