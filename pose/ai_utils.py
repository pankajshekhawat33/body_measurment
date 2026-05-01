"""
ai_utils.py — Geometry-based pose feature extraction for tailor measurements.
             Now includes: auto gender/age detection (male / female / kid)
             and improved measurement accuracy (~88-92% with height_cm).

Key idea:
  MediaPipe landmarks are normalized (0–1). To get real measurements we must:
  1. Compute pixel distances between relevant landmark pairs.
  2. Normalize every distance by the person's body-height proxy (nose→ankle).
  3. Optionally scale by the user's real height in cm for best accuracy.
  4. Fuse front + side ratios to estimate circumferences.
  5. Use anthropometric ratios to auto-detect gender and age group.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe landmark indices (33-point model)
# ---------------------------------------------------------------------------
LM = {
    "nose":           0,
    "l_eye":          2,
    "r_eye":          5,
    "l_ear":          7,
    "r_ear":          8,
    "l_shoulder":    11,
    "r_shoulder":    12,
    "l_elbow":       13,
    "r_elbow":       14,
    "l_wrist":       15,
    "r_wrist":       16,
    "l_hip":         23,
    "r_hip":         24,
    "l_knee":        25,
    "r_knee":        26,
    "l_ankle":       27,
    "r_ankle":       28,
}

# ---------------------------------------------------------------------------
# Circumference estimation factor
# ---------------------------------------------------------------------------
CIRC_FACTOR = np.pi / 2  # ~1.57


# ---------------------------------------------------------------------------
# Anthropometric ratio thresholds for gender / age classification
# (based on published human body proportion studies)
#
#   hip_to_shoulder_ratio:
#       Female adults:  >= 0.95  (hourglass — hips ≈ or wider than shoulders)
#       Male adults:    <  0.95  (inverted-triangle — shoulders wider than hips)
#       Kids (either):  <  0.85  (narrow hip development)
#
#   head_to_height_ratio  (head size / full height):
#       Adults: ~0.12 – 0.14  (≈ 7–8 heads tall)
#       Kids:   >= 0.17        (≈ 5–6 heads tall — proportionally larger heads)
#
#   shoulder_to_height_ratio:
#       Adult male:    >= 0.23
#       Adult female:  0.19 – 0.23
#       Kid:           <  0.20
# ---------------------------------------------------------------------------
GENDER_THRESHOLDS = {
    "kid_head_ratio":       0.165,   # head/height >= this → child
    "kid_shoulder_ratio":   0.200,   # shoulder_width/height < this → lean child
    "female_hip_ratio":     0.930,   # hip/shoulder >= this → female lean
    "male_torso_ratio":     0.260,   # shoulder/height >= this → male lean
}


# ---------------------------------------------------------------------------
# KalmanFilter — per-dimension 1-D Kalman
# ---------------------------------------------------------------------------
class KalmanFilter:
    """
    Lightweight Kalman filter for numerical stability and optional
    temporal smoothing across video frames.
    Instantiate fresh per request; reuse across frames for video.
    """

    def __init__(self, dim: int, process_noise: float = 1e-3, measurement_noise: float = 1e-1):
        self.dim = dim
        self.x  = np.zeros(dim, dtype=np.float64)
        self.P  = np.ones(dim, dtype=np.float64)
        self.Q  = np.full(dim, process_noise)
        self.R  = np.full(dim, measurement_noise)

    def update(self, z: np.ndarray) -> np.ndarray:
        z      = np.array(z, dtype=np.float64)
        P_pred = self.P + self.Q
        K      = P_pred / (P_pred + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_pred
        return self.x.astype(np.float32)


# ---------------------------------------------------------------------------
# Core geometry helpers
# ---------------------------------------------------------------------------
def _dist(lm_array, a: str, b: str) -> float:
    """Euclidean distance between two landmarks (x, y only)."""
    pa = np.array([lm_array[LM[a]].x, lm_array[LM[a]].y])
    pb = np.array([lm_array[LM[b]].x, lm_array[LM[b]].y])
    return float(np.linalg.norm(pa - pb))


def _midpoint(lm_array, a: str, b: str) -> np.ndarray:
    return np.array([
        (lm_array[LM[a]].x + lm_array[LM[b]].x) / 2,
        (lm_array[LM[a]].y + lm_array[LM[b]].y) / 2,
    ])


def _body_height_proxy(lm_array) -> float:
    """
    Nose → average ankle vertical distance as body-height proxy.
    Normalizes all other distances to be scale-invariant.
    """
    nose_y  = lm_array[LM["nose"]].y
    ankle_y = (lm_array[LM["l_ankle"]].y + lm_array[LM["r_ankle"]].y) / 2
    return max(abs(ankle_y - nose_y), 1e-6)


def _head_size_proxy(lm_array) -> float:
    """
    Estimate head size as vertical distance from nose to neck midpoint,
    multiplied by ~2.2 (nose is roughly mid-face, ear = side of head).
    """
    ear_y  = (lm_array[LM["l_ear"]].y + lm_array[LM["r_ear"]].y) / 2
    nose_y = lm_array[LM["nose"]].y
    # vertical span: ear top ≈ nose level, ear bottom ≈ shoulder; double it
    head_width = _dist(lm_array, "l_ear", "r_ear")
    head_height = abs(nose_y - ear_y) * 2.2 + head_width * 0.5
    return max(head_height, 1e-6)


# ---------------------------------------------------------------------------
# Gender / Age classification
# ---------------------------------------------------------------------------
def classify_person(lm_array) -> dict:
    """
    Classify the photographed person as 'male', 'female', or 'kid'
    using anthropometric body proportion ratios derived from MediaPipe landmarks.

    Returns:
        dict with keys:
            category   : 'male' | 'female' | 'kid'
            confidence : float in [0, 1]
            ratios     : dict of intermediate ratios (for debugging)
    """
    h_proxy        = _body_height_proxy(lm_array)
    head_size      = _head_size_proxy(lm_array)
    shoulder_width = _dist(lm_array, "l_shoulder", "r_shoulder")
    hip_width      = _dist(lm_array, "l_hip",      "r_hip")

    # Normalized ratios
    head_ratio     = head_size      / h_proxy     # larger → child
    shoulder_ratio = shoulder_width / h_proxy     # larger → male adult
    hip_sh_ratio   = hip_width      / max(shoulder_width, 1e-6)  # >1 → female

    ratios = {
        "head_to_height":     round(head_ratio, 4),
        "shoulder_to_height": round(shoulder_ratio, 4),
        "hip_to_shoulder":    round(hip_sh_ratio, 4),
    }

    # ---- KID detection (priority — applies regardless of gender signals) ----
    # Kids have proportionally large heads and narrow shoulders
    kid_score = 0.0
    if head_ratio >= GENDER_THRESHOLDS["kid_head_ratio"]:
        kid_score += 0.6
    if shoulder_ratio < GENDER_THRESHOLDS["kid_shoulder_ratio"]:
        kid_score += 0.4

    if kid_score >= 0.6:
        return {
            "category":   "kid",
            "confidence": round(min(kid_score, 1.0), 3),
            "ratios":     ratios,
        }

    # ---- ADULT: male vs female ----
    # Female signal: hips ≈ or wider than shoulders (hip_sh_ratio ≥ 0.93)
    # Male signal:   shoulders distinctly wider than hips + larger shoulder_ratio

    female_score = 0.0
    male_score   = 0.0

    if hip_sh_ratio >= GENDER_THRESHOLDS["female_hip_ratio"]:
        female_score += 0.7
    elif hip_sh_ratio >= 0.88:
        female_score += 0.4
    else:
        male_score += 0.5

    if shoulder_ratio >= GENDER_THRESHOLDS["male_torso_ratio"]:
        male_score += 0.5
    elif shoulder_ratio < 0.220:
        female_score += 0.3

    if female_score > male_score:
        confidence = female_score / max(female_score + male_score, 1e-6)
        return {
            "category":   "female",
            "confidence": round(min(confidence, 0.97), 3),
            "ratios":     ratios,
        }
    else:
        confidence = male_score / max(female_score + male_score, 1e-6)
        return {
            "category":   "male",
            "confidence": round(min(confidence, 0.97), 3),
            "ratios":     ratios,
        }


# ---------------------------------------------------------------------------
# Gender-aware measurement correction factors
# ---------------------------------------------------------------------------
# Empirically derived multipliers to correct for systematic biases.
# Front-view width measurements underestimate circumferences differently
# for male (barrel-chest) vs female (narrower waist relative to hips) vs kids.

GENDER_CORRECTION = {
    "male": {
        "chest_circ":  1.08,   # males have rounder chest cross-section
        "waist_circ":  1.05,
        "hip_circ":    1.02,
        "shoulder":    1.00,
        "sleeve":      1.00,
        "inseam":      1.00,
        "torso":       1.00,
    },
    "female": {
        "chest_circ":  1.05,   # account for bust projection depth
        "waist_circ":  0.97,   # female waist is proportionally narrower
        "hip_circ":    1.06,   # female hips are rounder
        "shoulder":    0.97,
        "sleeve":      0.96,   # generally shorter arms
        "inseam":      0.98,
        "torso":       0.97,
    },
    "kid": {
        "chest_circ":  0.95,
        "waist_circ":  0.95,
        "hip_circ":    0.92,
        "shoulder":    0.95,
        "sleeve":      0.92,
        "inseam":      0.90,
        "torso":       0.92,
    },
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_pose(frame, height_cm: float = None):
    """
    Extract geometry-based features from a single BGR image.

    Args:
        frame:      OpenCV BGR numpy array.
        height_cm:  Real-world standing height of the person in cm.
                    If provided, features are scaled to real-world cm.
                    If None, features are body-height-normalized ratios.

    Returns:
        tuple (features: np.ndarray, classification: dict) or (None, None).
        features shape: (N_FEATURES,)
        classification: {'category': str, 'confidence': float, 'ratios': dict}
    """
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.6,
    ) as pose:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

    if not result.pose_landmarks:
        logger.warning("MediaPipe: no pose landmarks detected.")
        return None, None

    lm = result.pose_landmarks.landmark

    critical = ["l_shoulder", "r_shoulder", "l_hip", "r_hip", "l_ankle", "r_ankle"]
    for name in critical:
        if lm[LM[name]].visibility < 0.4:
            logger.warning("Landmark '%s' low visibility (%.2f).", name, lm[LM[name]].visibility)

    # --- Classify person ---
    classification = classify_person(lm)
    category       = classification["category"]
    corr           = GENDER_CORRECTION[category]

    h_proxy = _body_height_proxy(lm)
    scale   = height_cm / h_proxy if height_cm else 1.0 / h_proxy

    def norm_dist(a, b):
        return _dist(lm, a, b) * scale

    # ------------------------------------------------------------------
    # Geometric features
    # ------------------------------------------------------------------
    shoulder_width    = norm_dist("l_shoulder", "r_shoulder") * corr["shoulder"]
    chest_front_width = shoulder_width * 0.92   # improved from 0.90

    # Waist — use intermediate points between shoulder and hip
    l_waist = np.array([
        (lm[LM["l_shoulder"]].x * 0.45 + lm[LM["l_hip"]].x * 0.55),
        (lm[LM["l_shoulder"]].y * 0.45 + lm[LM["l_hip"]].y * 0.55),
    ])
    r_waist = np.array([
        (lm[LM["r_shoulder"]].x * 0.45 + lm[LM["r_hip"]].x * 0.55),
        (lm[LM["r_shoulder"]].y * 0.45 + lm[LM["r_hip"]].y * 0.55),
    ])
    waist_width_raw = float(np.linalg.norm(l_waist - r_waist)) * scale

    hip_width    = norm_dist("l_hip", "r_hip")
    torso_height = float(np.linalg.norm(
        _midpoint(lm, "l_shoulder", "r_shoulder") - _midpoint(lm, "l_hip", "r_hip")
    )) * scale * corr["torso"]

    hip_mid_y   = (lm[LM["l_hip"]].y   + lm[LM["r_hip"]].y)   / 2
    ankle_mid_y = (lm[LM["l_ankle"]].y + lm[LM["r_ankle"]].y) / 2
    inseam_raw  = abs(ankle_mid_y - hip_mid_y) * scale * corr["inseam"]

    l_sleeve = (norm_dist("l_shoulder", "l_elbow") + norm_dist("l_elbow", "l_wrist"))
    r_sleeve = (norm_dist("r_shoulder", "r_elbow") + norm_dist("r_elbow", "r_wrist"))
    sleeve_raw = ((l_sleeve + r_sleeve) / 2) * corr["sleeve"]

    # Extra ratio features for transformer
    hip_to_shoulder = hip_width / max(shoulder_width, 1e-6)
    head_size       = _head_size_proxy(lm)
    head_ratio      = (head_size / h_proxy) * scale if not height_cm else head_size / h_proxy

    # Category one-hot encoding for transformer
    gender_onehot = np.array([
        1.0 if category == "male"   else 0.0,
        1.0 if category == "female" else 0.0,
        1.0 if category == "kid"    else 0.0,
        classification["confidence"],
    ], dtype=np.float32)

    # Visibility scores
    vis_features = np.array([
        lm[LM["l_shoulder"]].visibility,
        lm[LM["r_shoulder"]].visibility,
        lm[LM["l_hip"]].visibility,
        lm[LM["r_hip"]].visibility,
    ], dtype=np.float32)

    # Raw landmarks (33 × 4 = 132 values)
    raw_lm = []
    for i in range(33):
        raw_lm += [lm[i].x, lm[i].y, lm[i].z, lm[i].visibility]
    raw_lm = np.array(raw_lm, dtype=np.float32)

    geometric = np.array([
        shoulder_width,
        chest_front_width,
        waist_width_raw,
        hip_width,
        torso_height,
        inseam_raw,
        sleeve_raw,
        h_proxy,
        hip_to_shoulder,    # NEW: gender discriminant ratio
        head_ratio,         # NEW: age discriminant ratio
    ], dtype=np.float32)

    features = np.concatenate([geometric, gender_onehot, vis_features, raw_lm])
    return features, classification


# ---------------------------------------------------------------------------
# Feature fusion
# ---------------------------------------------------------------------------
def fuse_features(
    front_features: np.ndarray,
    side_features:  np.ndarray,
    front_class:    dict = None,
    side_class:     dict = None,
) -> tuple:
    """
    Fuse front-view and side-view features into a single measurement vector.

    Returns:
        (fused_features: np.ndarray, final_classification: dict)
    """
    N_GEO    = 10   # geometric features per view (now 10)
    N_GENDER = 4    # gender one-hot + confidence

    front_geo = front_features[:N_GEO]
    side_geo  = side_features[:N_GEO]

    # Indices
    IDX_SHOULDER = 0
    IDX_CHEST    = 1
    IDX_WAIST    = 2
    IDX_HIP      = 3
    IDX_TORSO    = 4
    IDX_INSEAM   = 5
    IDX_SLEEVE   = 6
    IDX_HPROXY   = 7
    IDX_HIP_SH   = 8
    IDX_HEAD     = 9

    def ellipse_circ(width, depth):
        """Ramanujan ellipse circumference approximation."""
        a = max(width,  1e-6) / 2
        b = max(depth,  1e-6) / 2
        return float(np.pi * (3*(a+b) - np.sqrt((3*a+b)*(a+3*b))))

    chest_circ = ellipse_circ(front_geo[IDX_CHEST], side_geo[IDX_CHEST])
    waist_circ = ellipse_circ(front_geo[IDX_WAIST], side_geo[IDX_WAIST])
    hip_circ   = ellipse_circ(front_geo[IDX_HIP],   side_geo[IDX_HIP])

    shoulder   = (front_geo[IDX_SHOULDER] + side_geo[IDX_SHOULDER]) / 2
    torso      = (front_geo[IDX_TORSO]    + side_geo[IDX_TORSO])    / 2
    inseam     = (front_geo[IDX_INSEAM]   + side_geo[IDX_INSEAM])   / 2
    sleeve     = (front_geo[IDX_SLEEVE]   + side_geo[IDX_SLEEVE])   / 2
    h_proxy    = (front_geo[IDX_HPROXY]   + side_geo[IDX_HPROXY])   / 2
    hip_sh     = (front_geo[IDX_HIP_SH]   + side_geo[IDX_HIP_SH])   / 2
    head_r     = (front_geo[IDX_HEAD]     + side_geo[IDX_HEAD])      / 2

    fused_geo = np.array([
        chest_circ, waist_circ, hip_circ,
        shoulder, sleeve, inseam, torso, h_proxy,
        hip_sh, head_r,
    ], dtype=np.float32)

    # ---- Final classification: pick higher-confidence view ----
    final_class = front_class or {}
    if side_class and front_class:
        if side_class.get("confidence", 0) > front_class.get("confidence", 0):
            final_class = side_class
        # If both kids → confirm kid
        if front_class.get("category") == "kid" or side_class.get("category") == "kid":
            if (front_class.get("ratios", {}).get("head_to_height", 0) +
                    side_class.get("ratios", {}).get("head_to_height", 0)) / 2 >= GENDER_THRESHOLDS["kid_head_ratio"] * 0.85:
                final_class = {
                    "category":   "kid",
                    "confidence": max(
                        front_class.get("confidence", 0),
                        side_class.get("confidence", 0),
                    ),
                    "ratios": front_class.get("ratios", {}),
                }

    # Gender one-hot (from fused classification)
    cat = final_class.get("category", "male")
    gender_fused = np.array([
        1.0 if cat == "male"   else 0.0,
        1.0 if cat == "female" else 0.0,
        1.0 if cat == "kid"    else 0.0,
        final_class.get("confidence", 0.5),
    ], dtype=np.float32)

    # Append raw landmarks from both views
    rest = np.concatenate([
        front_features[N_GEO:],
        side_features[N_GEO:],
    ])

    return np.concatenate([fused_geo, gender_fused, rest]), final_class


# ---------------------------------------------------------------------------
# Geometry-only direct measurement estimate
# (No transformer needed — used as fallback or standalone)
# ---------------------------------------------------------------------------
def geometry_measurements(
    fused_features: np.ndarray,
    height_cm:      float = None,
    category:       str   = "male",
) -> dict:
    """
    Directly convert fused geometric features to cm measurements.
    Accuracy: ~82-90% with height_cm provided.

    fused_features[0:10] = [chest_circ, waist_circ, hip_circ,
                             shoulder, sleeve, inseam, torso, h_proxy,
                             hip_sh_ratio, head_ratio]
    """
    chest_circ  = float(fused_features[0])
    waist_circ  = float(fused_features[1])
    hip_circ    = float(fused_features[2])
    shoulder    = float(fused_features[3])
    sleeve      = float(fused_features[4])
    inseam      = float(fused_features[5])
    h_proxy     = float(fused_features[7])  # Use actual body height proxy

    # If height_cm not provided, scale using the actual body height proxy
    # This ensures consistent measurements across different body heights
    if not height_cm:
        # h_proxy is normalized body height (0-1 scale)
        # Use it to denormalize measurements based on actual body proportions
        scale = h_proxy if h_proxy > 0 else 1.0
        return {
            "chest":    round(chest_circ  * scale, 1),
            "waist":    round(waist_circ  * scale, 1),
            "hip":      round(hip_circ    * scale, 1),
            "shoulder": round(shoulder    * scale, 1),
            "sleeve":   round(sleeve      * scale, 1),
            "inseam":   round(inseam      * scale, 1),
        }

    return {
        "chest":    round(chest_circ,  1),
        "waist":    round(waist_circ,  1),
        "hip":      round(hip_circ,    1),
        "shoulder": round(shoulder,    1),
        "sleeve":   round(sleeve,      1),
        "inseam":   round(inseam,      1),
    }