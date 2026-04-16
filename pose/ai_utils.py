"""
ai_utils.py — Geometry-based pose feature extraction for tailor measurements.

Key idea:
  MediaPipe landmarks are normalized (0–1). To get real measurements we must:
  1. Compute pixel distances between relevant landmark pairs.
  2. Normalize every distance by the person's body-height proxy (nose→ankle).
  3. Optionally scale by the user's real height in cm for best accuracy.
  4. Fuse front + side ratios to estimate circumferences.

This replaces the raw concatenation approach which loses all size information.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe landmark indices (33-point model)
# ---------------------------------------------------------------------------
LM = {
    "nose":           0,
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
# When we measure width (diameter-like) from front, multiply by pi/2 to
# estimate half-circumference, then add the side depth similarly.
# This is an anthropometric approximation.
# ---------------------------------------------------------------------------
CIRC_FACTOR = np.pi / 2  # ~1.57


# ---------------------------------------------------------------------------
# KalmanFilter — proper 1-D Kalman per dimension
# ---------------------------------------------------------------------------
class KalmanFilter:
    """
    Per-request lightweight Kalman filter.
    For a single static image pair, this is used for numerical stability —
    not temporal smoothing. Instantiate fresh for every request.

    If you process a video stream (multiple frames), pass successive frames
    through the same instance to get temporal smoothing.
    """

    def __init__(self, dim: int, process_noise: float = 1e-3, measurement_noise: float = 1e-1):
        self.dim = dim
        self.x  = np.zeros(dim, dtype=np.float64)   # state estimate
        self.P  = np.ones(dim, dtype=np.float64)    # estimate covariance
        self.Q  = np.full(dim, process_noise)        # process noise
        self.R  = np.full(dim, measurement_noise)    # measurement noise

    def update(self, z: np.ndarray) -> np.ndarray:
        """Kalman update step. z = new measurement vector."""
        z = np.array(z, dtype=np.float64)
        # Predict
        P_pred = self.P + self.Q
        # Update
        K      = P_pred / (P_pred + self.R)          # Kalman gain
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_pred
        return self.x.astype(np.float32)


# ---------------------------------------------------------------------------
# Core geometry helpers
# ---------------------------------------------------------------------------
def _dist(lm_array, a: str, b: str) -> float:
    """Euclidean distance between two landmarks (using x, y only)."""
    pa = np.array([lm_array[LM[a]].x, lm_array[LM[a]].y])
    pb = np.array([lm_array[LM[b]].x, lm_array[LM[b]].y])
    return float(np.linalg.norm(pa - pb))


def _midpoint(lm_array, a: str, b: str):
    return np.array([
        (lm_array[LM[a]].x + lm_array[LM[b]].x) / 2,
        (lm_array[LM[a]].y + lm_array[LM[b]].y) / 2,
    ])


def _body_height_proxy(lm_array) -> float:
    """
    Nose → average ankle vertical distance as a proxy for standing height.
    Used to normalize all other distances so they are scale-invariant.
    """
    nose_y    = lm_array[LM["nose"]].y
    ankle_y   = (lm_array[LM["l_ankle"]].y + lm_array[LM["r_ankle"]].y) / 2
    height    = abs(ankle_y - nose_y)
    return max(height, 1e-6)  # avoid division by zero


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
                    If None, features are body-height-normalized ratios (0–1 range).

    Returns:
        np.ndarray of shape (N_FEATURES,) or None if pose not detected.
    """
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose

    # Use static_image_mode=True for single photos (better accuracy than video mode)
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,          # 0=fast, 1=default, 2=most accurate
        enable_segmentation=False,
        min_detection_confidence=0.6,
    ) as pose:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

    if not result.pose_landmarks:
        logger.warning("MediaPipe: no pose landmarks detected.")
        return None

    lm = result.pose_landmarks.landmark

    # Check that critical landmarks are visible enough
    critical = ["l_shoulder", "r_shoulder", "l_hip", "r_hip", "l_ankle", "r_ankle"]
    for name in critical:
        if lm[LM[name]].visibility < 0.4:
            logger.warning("Landmark '%s' has low visibility (%.2f).", name, lm[LM[name]].visibility)
            # Don't return None — partial visibility still gives useful signal

    h_proxy = _body_height_proxy(lm)
    scale   = height_cm / h_proxy if height_cm else 1.0 / h_proxy

    def norm_dist(a, b):
        return _dist(lm, a, b) * scale

    # ------------------------------------------------------------------
    # Geometric features (all in cm if height_cm given, else normalized)
    # ------------------------------------------------------------------

    # --- Shoulder ---
    shoulder_width = norm_dist("l_shoulder", "r_shoulder")

    # --- Chest (front width proxy — shoulder-to-shoulder at chest level) ---
    # Chest is approximately 90% of shoulder width at mid-torso
    chest_front_width = shoulder_width * 0.90

    # --- Waist (distance between hip-shoulder midpoint) ---
    l_waist_y = (lm[LM["l_shoulder"]].y + lm[LM["l_hip"]].y) / 2
    r_waist_y = (lm[LM["r_shoulder"]].y + lm[LM["r_hip"]].y) / 2
    l_waist_x = (lm[LM["l_shoulder"]].x + lm[LM["l_hip"]].x) / 2
    r_waist_x = (lm[LM["r_shoulder"]].x + lm[LM["r_hip"]].x) / 2
    waist_width_raw = np.linalg.norm(
        np.array([l_waist_x, l_waist_y]) - np.array([r_waist_x, r_waist_y])
    ) * scale

    # --- Hip ---
    hip_width = norm_dist("l_hip", "r_hip")

    # --- Torso height (shoulder midpoint → hip midpoint) ---
    torso_height = np.linalg.norm(
        _midpoint(lm, "l_shoulder", "r_shoulder") - _midpoint(lm, "l_hip", "r_hip")
    ) * scale

    # --- Inseam (hip midpoint → ankle midpoint, vertical) ---
    hip_mid_y   = (lm[LM["l_hip"]].y   + lm[LM["r_hip"]].y)   / 2
    ankle_mid_y = (lm[LM["l_ankle"]].y + lm[LM["r_ankle"]].y) / 2
    inseam_raw  = abs(ankle_mid_y - hip_mid_y) * scale

    # --- Sleeve (shoulder → wrist along arm) ---
    l_sleeve = norm_dist("l_shoulder", "l_elbow") + norm_dist("l_elbow", "l_wrist")
    r_sleeve = norm_dist("r_shoulder", "r_elbow") + norm_dist("r_elbow", "r_wrist")
    sleeve_raw = (l_sleeve + r_sleeve) / 2

    # --- Visibility scores for critical points (confidence signal) ---
    vis_features = np.array([
        lm[LM["l_shoulder"]].visibility,
        lm[LM["r_shoulder"]].visibility,
        lm[LM["l_hip"]].visibility,
        lm[LM["r_hip"]].visibility,
    ], dtype=np.float32)

    # --- Raw landmark array (normalized) for transformer fine-tuning ---
    raw_lm = []
    for i in range(33):
        raw_lm += [lm[i].x, lm[i].y, lm[i].z, lm[i].visibility]
    raw_lm = np.array(raw_lm, dtype=np.float32)

    # --- Combine all features ---
    geometric = np.array([
        shoulder_width,
        chest_front_width,
        waist_width_raw,
        hip_width,
        torso_height,
        inseam_raw,
        sleeve_raw,
        h_proxy,         # body height proxy (useful for the transformer)
    ], dtype=np.float32)

    features = np.concatenate([geometric, vis_features, raw_lm])
    return features


# ---------------------------------------------------------------------------
# Feature fusion
# ---------------------------------------------------------------------------
def fuse_features(front_features: np.ndarray, side_features: np.ndarray) -> np.ndarray:
    """
    Fuse front-view and side-view features into a single measurement vector.

    Strategy:
      - Geometric features (first 8): front gives width, side gives depth.
        We combine them to estimate circumferences.
      - Visibility + raw landmarks: simply concatenate for transformer input.

    Returns:
        np.ndarray of shape (front.shape[0] + side.shape[0],)
    """
    N_GEO = 8   # number of geometric features at the start of each vector
    N_VIS = 4   # visibility features

    front_geo = front_features[:N_GEO]
    side_geo  = side_features[:N_GEO]

    # Index mapping for geometric features
    # [shoulder_width, chest_front_width, waist_width, hip_width,
    #  torso_height, inseam, sleeve, h_proxy]
    IDX_SHOULDER = 0
    IDX_CHEST    = 1
    IDX_WAIST    = 2
    IDX_HIP      = 3
    IDX_TORSO    = 4
    IDX_INSEAM   = 5
    IDX_SLEEVE   = 6
    IDX_HPROXY   = 7

    # Circumference estimation:
    # model: body cross-section ≈ ellipse
    # circumference ≈ pi * (3(a+b) - sqrt((3a+b)(a+3b))) / 2  [Ramanujan approx]
    # simplified: C ≈ pi/2 * (width + depth) for near-circular sections
    def ellipse_circ(width, depth):
        a = width / 2
        b = depth / 2
        return np.pi * (3*(a+b) - np.sqrt((3*a+b)*(a+3*b)))

    chest_circ  = ellipse_circ(front_geo[IDX_CHEST],    side_geo[IDX_CHEST])
    waist_circ  = ellipse_circ(front_geo[IDX_WAIST],    side_geo[IDX_WAIST])
    hip_circ    = ellipse_circ(front_geo[IDX_HIP],      side_geo[IDX_HIP])

    # Length-based measurements: average front+side (should be similar)
    shoulder    = (front_geo[IDX_SHOULDER] + side_geo[IDX_SHOULDER]) / 2
    torso       = (front_geo[IDX_TORSO]   + side_geo[IDX_TORSO])    / 2
    inseam      = (front_geo[IDX_INSEAM]  + side_geo[IDX_INSEAM])   / 2
    sleeve      = (front_geo[IDX_SLEEVE]  + side_geo[IDX_SLEEVE])   / 2
    h_proxy     = (front_geo[IDX_HPROXY]  + side_geo[IDX_HPROXY])   / 2

    # Fused geometric block
    fused_geo = np.array([
        chest_circ,
        waist_circ,
        hip_circ,
        shoulder,
        sleeve,
        inseam,
        torso,
        h_proxy,
    ], dtype=np.float32)

    # Append raw landmarks from both views for transformer
    rest = np.concatenate([front_features[N_GEO:], side_features[N_GEO:]])

    return np.concatenate([fused_geo, rest])