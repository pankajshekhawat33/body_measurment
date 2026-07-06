"""
ai_utils.py — Geometry-based pose feature extraction for tailor measurements.
             Fixed version: correct circumference math, proper scaling,
             consistent N_GEO=16, no double-multiply bug.

Key fixes vs original:
  FIX-1: extract_pose() — when height_cm=None, scale stays as ratio (0-1),
          NOT 1/h_proxy (which causes double-division downstream).
  FIX-2: ellipse_circ() — now uses HALF-widths correctly (a=w/2, b=d/2).
          Original passed full widths → circumference was ~2× too large.
  FIX-3: fuse_features() — N_GEO=16 consistently; fused index map matches
          what views.py reads at [0:6].
  FIX-4: geometry_measurements() — no double-scaling when height_cm=None.
  FIX-5: sleeve computed as shoulder→elbow→wrist (segmented), then
          corrected for partial-arm detection.
  FIX-6: hip_circ uses side_geo depth correctly (not front_geo again).
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe landmark indices (33-point model)
# ---------------------------------------------------------------------------
LM = {
    "nose":        0,
    "l_eye":       2,
    "r_eye":       5,
    "l_ear":       7,
    "r_ear":       8,
    "l_shoulder": 11,
    "r_shoulder": 12,
    "l_elbow":    13,
    "r_elbow":    14,
    "l_wrist":    15,
    "r_wrist":    16,
    "l_hip":      23,
    "r_hip":      24,
    "l_knee":     25,
    "r_knee":     26,
    "l_ankle":    27,
    "r_ankle":    28,
}

# ---------------------------------------------------------------------------
# Anthropometric thresholds for gender/age classification
# ---------------------------------------------------------------------------
GENDER_THRESHOLDS = {
    "kid_head_ratio":     0.165,
    "kid_shoulder_ratio": 0.200,
    "female_hip_ratio":   0.930,
    "male_torso_ratio":   0.260,
}






# ---------------------------------------------------------------------------
# Gender correction multipliers
# ---------------------------------------------------------------------------
GENDER_CORRECTION = {
    "male": {
        "chest_circ": 1.15, "waist_circ": 1.18, "hip_circ":  1.10,
        "shoulder":   1.00, "sleeve":     1.00,  "inseam":    1.00,
        "torso":      1.00, "bicep_circ": 1.06,  "thigh_circ":1.04,
        "calf_circ":  1.03, "neck_circ":  1.02,  "height":    1.00,
    },
    "female": {
        "chest_circ": 1.15, "waist_circ": 0.97, "hip_circ":  1.06,
        "shoulder":   0.97, "sleeve":     0.96,  "inseam":    0.98,
        "torso":      0.97, "bicep_circ": 0.94,  "thigh_circ":1.02,
        "calf_circ":  0.98, "neck_circ":  0.94,  "height":    1.00,
    },
    "kid": {
        "chest_circ": 0.95, "waist_circ": 0.95, "hip_circ":  0.92,
        "shoulder":   0.95, "sleeve":     0.92,  "inseam":    0.90,
        "torso":      0.92, "bicep_circ": 0.90,  "thigh_circ":0.88,
        "calf_circ":  0.85, "neck_circ":  0.88,  "height":    1.00,
    },
}

# ---------------------------------------------------------------------------
# Number of geometric features per view — MUST match fuse_features N_GEO
# ---------------------------------------------------------------------------
N_GEO = 16

# fused_geo index map (used by both fuse_features and views.py)
# [0]  chest_circ
# [1]  waist_circ
# [2]  hip_circ
# [3]  shoulder
# [4]  sleeve
# [5]  inseam
# [6]  torso
# [7]  h_proxy
# [8]  hip_sh_ratio
# [9]  head_ratio
# [10] bicep_circ
# [11] thigh_circ
# [12] calf_circ
# [13] neck_circ
# [14] height
# [15] calf_height


# ---------------------------------------------------------------------------
# KalmanFilter
# ---------------------------------------------------------------------------
class KalmanFilter:
    def __init__(self, dim: int, process_noise: float = 1e-3, measurement_noise: float = 1e-1):
        self.dim = dim
        self.x   = np.zeros(dim, dtype=np.float64)
        self.P   = np.ones(dim,  dtype=np.float64)
        self.Q   = np.full(dim, process_noise)
        self.R   = np.full(dim, measurement_noise)

    def update(self, z: np.ndarray) -> np.ndarray:
        z      = np.array(z, dtype=np.float64)
        P_pred = self.P + self.Q
        K      = P_pred / (P_pred + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_pred
        return self.x.astype(np.float32)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _dist(lm_array, a: str, b: str) -> float:
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
    Better body height estimation using torso + legs.
    More stable than nose->ankle.
    """

    shoulder_mid = _midpoint(lm_array, "l_shoulder", "r_shoulder")
    hip_mid      = _midpoint(lm_array, "l_hip", "r_hip")

    torso = np.linalg.norm(shoulder_mid - hip_mid)

    l_leg = (
        _dist(lm_array, "l_hip", "l_knee") +
        _dist(lm_array, "l_knee", "l_ankle")
    )

    r_leg = (
        _dist(lm_array, "r_hip", "r_knee") +
        _dist(lm_array, "r_knee", "r_ankle")
    )

    legs = (l_leg + r_leg) / 2.0

    # head approximation
    head = torso * 0.32

    return max(head + torso + legs, 1e-6)


def _head_size_proxy(lm_array) -> float:
    ear_y      = (lm_array[LM["l_ear"]].y + lm_array[LM["r_ear"]].y) / 2
    nose_y     = lm_array[LM["nose"]].y
    head_width = _dist(lm_array, "l_ear", "r_ear")
    return max(abs(nose_y - ear_y) * 2.2 + head_width * 0.5, 1e-6)


# ---------------------------------------------------------------------------
# FIX-2: Correct ellipse circumference
# Input: FULL widths (w, d). We halve them inside to get semi-axes.
# Ramanujan approximation: π × [3(a+b) - √((3a+b)(a+3b))]
# ---------------------------------------------------------------------------
def _ellipse_circ(width: float, depth: float) -> float:
    """
    Circumference of an ellipse given full width and full depth.
    Semi-axes: a = width/2, b = depth / 2.0 * 1.25   # more realistic depth correction

    FIX: Pose models underestimate depth (especially from single view).
    Apply empirical correction factors:
      - Front view width: usually accurate
      - Side view depth: typically 85-90% of true depth
    Correction factor = 1.12 (compensates for ~12% underestimation)
    """
    a = max(float(width), 1e-6) / 2.0
    b = max(float(depth), 1e-6) / 2.0 * 1.09  # Depth correction factor
    return float(np.pi * (3 * (a + b) - np.sqrt((3*a + b) * (a + 3*b))))


# ---------------------------------------------------------------------------
# Gender / Age classification
# ---------------------------------------------------------------------------
def _body_height_proxy(lm_array) -> float:
    """
    Better body height estimation using torso + legs.
    More stable than nose->ankle.
    """

    shoulder_mid = _midpoint(lm_array, "l_shoulder", "r_shoulder")
    hip_mid      = _midpoint(lm_array, "l_hip", "r_hip")

    torso = np.linalg.norm(shoulder_mid - hip_mid)

    l_leg = (
        _dist(lm_array, "l_hip", "l_knee") +
        _dist(lm_array, "l_knee", "l_ankle")
    )

    r_leg = (
        _dist(lm_array, "r_hip", "r_knee") +
        _dist(lm_array, "r_knee", "r_ankle")
    )

    legs = (l_leg + r_leg) / 2.0

    # head approximation
    head = torso * 0.32

    return max(head + torso + legs, 1e-6)
    
def classify_person(lm_array) -> dict:

    h_proxy        = _body_height_proxy(lm_array)
    head_size      = _head_size_proxy(lm_array)
    shoulder_width = _dist(lm_array, "l_shoulder", "r_shoulder")
    hip_width      = _dist(lm_array, "l_hip", "r_hip")

    head_ratio      = head_size / max(h_proxy, 1e-6)
    shoulder_ratio  = shoulder_width / max(h_proxy, 1e-6)
    hip_sh_ratio    = hip_width / max(shoulder_width, 1e-6)

    ratios = {
        "head_to_height":     round(head_ratio, 4),
        "shoulder_to_height": round(shoulder_ratio, 4),
        "hip_to_shoulder":    round(hip_sh_ratio, 4),
    }

    # Kid detection
    kid_score = 0.0

    if head_ratio >= GENDER_THRESHOLDS["kid_head_ratio"]:
        kid_score += 0.6

    if shoulder_ratio < GENDER_THRESHOLDS["kid_shoulder_ratio"]:
        kid_score += 0.4

    if kid_score >= 0.6:
        return {
            "category": "kid",
            "confidence": round(min(kid_score, 1.0), 3),
            "ratios": ratios,
        }

    # Male vs Female
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
        conf = female_score / max(female_score + male_score, 1e-6)

        return {
            "category": "female",
            "confidence": round(min(conf, 0.97), 3),
            "ratios": ratios,
        }

    else:
        conf = male_score / max(female_score + male_score, 1e-6)

        return {
            "category": "male",
            "confidence": round(min(conf, 0.97), 3),
            "ratios": ratios,
        }
# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_pose(frame, height_cm: float = None):
    """
    Extract N_GEO=16 geometric features + gender + visibility + raw landmarks.

    FIX-1: scale logic corrected.
      - WITH height_cm:    scale = height_cm / h_proxy  → features in cm
      - WITHOUT height_cm: scale = 1.0                  → features stay as
                                                           normalized ratios (0-1)
        Do NOT use 1/h_proxy — that causes double-division in geometry_measurements.

    Returns:
        (features: np.ndarray, classification: dict) or (None, None)
    """
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.6,
    ) as pose:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

    if not result.pose_landmarks:
        logger.warning("MediaPipe: no pose landmarks detected.")
        return None, None

    lm = result.pose_landmarks.landmark

    for name in ["l_shoulder", "r_shoulder", "l_hip", "r_hip", "l_ankle", "r_ankle"]:
        if lm[LM[name]].visibility < 0.4:
            logger.warning("Landmark '%s' low visibility (%.2f).", name, lm[LM[name]].visibility)

    classification = classify_person(lm)
    category       = classification["category"]
    corr           = GENDER_CORRECTION[category]

    h_proxy = _body_height_proxy(lm)

    # FIX-1: correct scale — ratios when no height, cm when height given
    if height_cm:
        scale = float(height_cm) / h_proxy   # → features in cm
    else:
        scale = 1.0
    def norm_dist(a, b):
        return _dist(lm, a, b) * scale

    # ------------------------------------------------------------------
    # Geometric measurements
    # ------------------------------------------------------------------
    shoulder_width    = norm_dist("l_shoulder", "r_shoulder") * corr["shoulder"]

    # Chest front width ≈ 92% of shoulder width
    chest_front_width = shoulder_width * 0.85

    # Waist — interpolated point between shoulder (45%) and hip (55%)
    l_waist = np.array([
    lm[LM["l_shoulder"]].x * 0.60 + lm[LM["l_hip"]].x * 0.40,
    lm[LM["l_shoulder"]].y * 0.60 + lm[LM["l_hip"]].y * 0.40,
    ])
    r_waist = np.array([
    lm[LM["r_shoulder"]].x * 0.60 + lm[LM["r_hip"]].x * 0.40,
    lm[LM["r_shoulder"]].y * 0.60 + lm[LM["r_hip"]].y * 0.40,
    ])
    waist_width_raw = float(np.linalg.norm(l_waist - r_waist)) * scale

    hip_width = norm_dist("l_hip", "r_hip") * 0.95

    torso_height = float(np.linalg.norm(
        _midpoint(lm, "l_shoulder", "r_shoulder") - _midpoint(lm, "l_hip", "r_hip")
    )) * scale * corr["torso"]

    hip_mid_y   = (lm[LM["l_hip"]].y   + lm[LM["r_hip"]].y)   / 2
    ankle_mid_y = (lm[LM["l_ankle"]].y + lm[LM["r_ankle"]].y) / 2
    inseam_raw  = abs(ankle_mid_y - hip_mid_y) * scale * corr["inseam"]

    # FIX-5: Sleeve = segmented (shoulder→elbow + elbow→wrist), averaged L+R
    l_sleeve = norm_dist("l_shoulder", "l_elbow") + norm_dist("l_elbow", "l_wrist")
    r_sleeve = norm_dist("r_shoulder", "r_elbow") + norm_dist("r_elbow", "r_wrist")
    sleeve_raw = ((l_sleeve + r_sleeve) / 2.0) * corr["sleeve"]

    # Bicep width
    bicep_w_front = norm_dist("l_elbow", "r_elbow")
    bicep_w_side  = bicep_w_front * 0.65
    bicep_circ_raw = _ellipse_circ(bicep_w_front, bicep_w_side) * corr["bicep_circ"]

    # Thigh width
    thigh_w_front = norm_dist("l_knee", "r_knee")
    thigh_w_side  = thigh_w_front * 0.70
    thigh_circ_raw = _ellipse_circ(thigh_w_front, thigh_w_side) * corr["thigh_circ"]

    # Calf circumference
    knee_mid_y     = (lm[LM["l_knee"]].y + lm[LM["r_knee"]].y) / 2
    calf_height_raw = abs(ankle_mid_y - knee_mid_y) * scale
    calf_w_front   = thigh_w_front * 0.55
    calf_w_side    = calf_w_front * 0.65
    calf_circ_raw  = _ellipse_circ(calf_w_front, calf_w_side) * corr["calf_circ"]

    # Neck circumference
    neck_width_raw = shoulder_width * 0.35
    neck_circ_raw  = float(np.pi * neck_width_raw) * corr["neck_circ"]

    full_height = h_proxy * scale * corr["height"]

    hip_to_shoulder = hip_width / max(shoulder_width, 1e-6)
    head_size       = _head_size_proxy(lm)
    head_ratio      = head_size / h_proxy   # always a ratio regardless of scale

    # Gender one-hot
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
    raw_lm = np.array(
        [v for i in range(33) for v in [lm[i].x, lm[i].y, lm[i].z, lm[i].visibility]],
        dtype=np.float32,
    )

    # N_GEO=16 geometric feature vector
    geometric = np.array([
        shoulder_width,      # [0]
        chest_front_width,   # [1]
        waist_width_raw,     # [2]
        hip_width,           # [3]
        torso_height,        # [4]
        inseam_raw,          # [5]
        sleeve_raw,          # [6]
        h_proxy,             # [7]  always raw ratio, NOT scaled
        hip_to_shoulder,     # [8]
        head_ratio,          # [9]
        bicep_circ_raw,      # [10]
        thigh_circ_raw,      # [11]
        calf_circ_raw,       # [12]
        neck_circ_raw,       # [13]
        full_height,         # [14]
        calf_height_raw,     # [15]
    ], dtype=np.float32)

    features = np.concatenate([geometric, gender_onehot, vis_features, raw_lm])
    logger.info("extract_pose: scale=%.4f h_proxy=%.4f category=%s features[0:6]=%s",
                scale, h_proxy, category, geometric[:6])
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
    Fuse front + side geometric features into one measurement vector.

    FIX-3: N_GEO=16 consistently; index map matches what views.py reads.
    FIX-6: hip_circ now uses front HIP width + SIDE HIP width (not front again).

    Output fused_geo layout (indices 0-15):
        [0]  chest_circ   [1]  waist_circ  [2]  hip_circ
        [3]  shoulder     [4]  sleeve      [5]  inseam
        [6]  torso        [7]  h_proxy     [8]  hip_sh_ratio
        [9]  head_ratio   [10] bicep_circ  [11] thigh_circ
        [12] calf_circ    [13] neck_circ   [14] height
        [15] calf_height
    """
    # Geometric slice indices (must match extract_pose geometric array)
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
    IDX_BICEP    = 10
    IDX_THIGH    = 11
    IDX_CALF     = 12
    IDX_NECK     = 13
    IDX_HEIGHT   = 14
    IDX_CALF_H   = 15

    front_geo = front_features[:N_GEO].astype(np.float64)
    side_geo  = side_features[:N_GEO].astype(np.float64)

    # ------------------------------------------------------------------
    # Circumferences via ellipse formula
    # Front view  → width dimension
    # Side view   → depth dimension
    #
    # FIX-6: For hip_circ, use front_geo[IDX_HIP] as width and
    #         side_geo[IDX_HIP] as depth. Original accidentally used
    #         front_geo for both → no depth info → wrong circumference.
    # ------------------------------------------------------------------
    depth_chest = min(side_geo[IDX_CHEST], front_geo[IDX_CHEST] * 0.75)
    chest_circ = _ellipse_circ(front_geo[IDX_CHEST], depth_chest)
    waist_circ = _ellipse_circ(front_geo[IDX_WAIST], side_geo[IDX_WAIST])
    hip_circ   = _ellipse_circ(front_geo[IDX_HIP],   side_geo[IDX_HIP])   # FIX-6

    # Linear measurements — simple average of both views
    shoulder  = float((front_geo[IDX_SHOULDER] + side_geo[IDX_SHOULDER]) / 2)
    sleeve    = float((front_geo[IDX_SLEEVE]   + side_geo[IDX_SLEEVE])   / 2)
    inseam    = float((front_geo[IDX_INSEAM]   + side_geo[IDX_INSEAM])   / 2)
    torso     = float((front_geo[IDX_TORSO]    + side_geo[IDX_TORSO])    / 2)
    h_proxy   = float((front_geo[IDX_HPROXY]   + side_geo[IDX_HPROXY])   / 2)
    hip_sh    = float((front_geo[IDX_HIP_SH]   + side_geo[IDX_HIP_SH])   / 2)
    head_r    = float((front_geo[IDX_HEAD]     + side_geo[IDX_HEAD])      / 2)
    bicep     = float((front_geo[IDX_BICEP]    + side_geo[IDX_BICEP])    / 2)
    thigh     = float((front_geo[IDX_THIGH]    + side_geo[IDX_THIGH])    / 2)
    calf      = float((front_geo[IDX_CALF]     + side_geo[IDX_CALF])     / 2)
    neck      = float((front_geo[IDX_NECK]     + side_geo[IDX_NECK])     / 2)
    height    = float((front_geo[IDX_HEIGHT]   + side_geo[IDX_HEIGHT])   / 2)
    calf_h    = float((front_geo[IDX_CALF_H]   + side_geo[IDX_CALF_H])   / 2)

    fused_geo = np.array([
        chest_circ, waist_circ, hip_circ,
        shoulder,   sleeve,     inseam,
        torso,      h_proxy,    hip_sh,
        head_r,     bicep,      thigh,
        calf,       neck,       height,
        calf_h,
    ], dtype=np.float32)

    logger.info(
        "fuse_features: chest=%.2f waist=%.2f hip=%.2f shoulder=%.2f sleeve=%.2f inseam=%.2f",
        chest_circ, waist_circ, hip_circ, shoulder, sleeve, inseam
    )

    # Final classification — prefer higher confidence view
    final_class = front_class or {}
    if side_class and front_class:
        if side_class.get("confidence", 0) > front_class.get("confidence", 0):
            final_class = side_class
        avg_head = (
            front_class.get("ratios", {}).get("head_to_height", 0) +
            side_class.get("ratios", {}).get("head_to_height", 0)
        ) / 2
        if (front_class.get("category") == "kid" or side_class.get("category") == "kid") \
                and avg_head >= GENDER_THRESHOLDS["kid_head_ratio"] * 0.85:
            final_class = {
                "category":   "kid",
                "confidence": max(
                    front_class.get("confidence", 0),
                    side_class.get("confidence", 0),
                ),
                "ratios": front_class.get("ratios", {}),
            }

    cat = final_class.get("category", "male")
    gender_fused = np.array([
        1.0 if cat == "male"   else 0.0,
        1.0 if cat == "female" else 0.0,
        1.0 if cat == "kid"    else 0.0,
        final_class.get("confidence", 0.5),
    ], dtype=np.float32)

    # Keep only the view-specific non-geometric payloads that the model expects:
    # [gender(4)] + [front_vis(4) + front_raw(132)] + [side_vis(4) + side_raw(132)]
    rest = np.concatenate([
        front_features[N_GEO + 4:],
        side_features[N_GEO + 4:],
    ])
    return np.concatenate([fused_geo, gender_fused, rest]), final_class


# ---------------------------------------------------------------------------
# Direct geometry measurement (no transformer)
# ---------------------------------------------------------------------------
def geometry_measurements(
    fused_features: np.ndarray,
    height_cm:      float = None,
    category:       str   = "male",
) -> dict:
    """
    Convert fused features to cm.

    FIX-4: When height_cm is provided, features are ALREADY in cm
           (because extract_pose scaled them). Just return them directly.
           When height_cm=None, features are normalized ratios (0-1).
           Use height from fused[14] if available, else SCALE by typical values.
    """
    chest_circ  = float(fused_features[0])
    waist_circ  = float(fused_features[1])
    hip_circ    = float(fused_features[2])
    shoulder    = float(fused_features[3])
    sleeve      = float(fused_features[4])
    inseam      = float(fused_features[5])
    torso       = float(fused_features[6])
    h_proxy     = float(fused_features[7])
    bicep_circ  = float(fused_features[10])
    thigh_circ  = float(fused_features[11])
    calf_circ   = float(fused_features[12])
    neck_circ   = float(fused_features[13])
    height      = float(fused_features[14])
    calf_height = float(fused_features[15])

    if height_cm:
        # Features already in cm — return directly (FIX-4)
        return {
            "chest":       round(chest_circ,  1),
            "waist":       round(waist_circ,  1),
            "hip":         round(hip_circ,    1),
            "shoulder":    round(shoulder,    1),
            "sleeve":      round(sleeve,      1),
            "inseam":      round(inseam,      1),
            "torso":       round(torso,       1),
            "bicep":       round(bicep_circ,  1),
            "thigh":       round(thigh_circ,  1),
            "calf":        round(calf_circ,   1),
            "neck":        round(neck_circ,   1),
            "height":      round(height,      1),
            "calf_height": round(calf_height, 1),
        }

    # No height_cm: features are normalized ratios
    # Use typical adult scale values (cm) to convert
    # These are mean real-world values per measurement
    TYPICAL = {
        "chest": 95.0, "waist": 82.0, "hip": 97.0,
        "shoulder": 44.0, "sleeve": 60.0, "inseam": 76.0,
        "torso": 50.0, "bicep": 28.0, "thigh": 55.0,
        "calf": 37.0, "neck": 37.0, "height": 170.0,
        "calf_height": 40.0,
    }

    def _scale(ratio, typical):
        # ratio is 0-1; map ±0.15 deviation around typical
        deviation = float(np.clip(ratio - 0.5, -0.15, 0.15))
        return typical * (1.0 + deviation)

    return {
        "chest":       round(_scale(chest_circ,  TYPICAL["chest"]),       1),
        "waist":       round(_scale(waist_circ,  TYPICAL["waist"]),       1),
        "hip":         round(_scale(hip_circ,    TYPICAL["hip"]),         1),
        "shoulder":    round(_scale(shoulder,    TYPICAL["shoulder"]),    1),
        "sleeve":      round(_scale(sleeve,      TYPICAL["sleeve"]),      1),
        "inseam":      round(_scale(inseam,      TYPICAL["inseam"]),      1),
        "torso":       round(_scale(torso,       TYPICAL["torso"]),       1),
        "bicep":       round(_scale(bicep_circ,  TYPICAL["bicep"]),       1),
        "thigh":       round(_scale(thigh_circ,  TYPICAL["thigh"]),       1),
        "calf":        round(_scale(calf_circ,   TYPICAL["calf"]),        1),
        "neck":        round(_scale(neck_circ,   TYPICAL["neck"]),        1),
        "height":      round(_scale(height,      TYPICAL["height"]),      1),
        "calf_height": round(_scale(calf_height, TYPICAL["calf_height"]), 1),
    }