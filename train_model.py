import os
import numpy as np
import pandas as pd
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
DATASET_PATH = "pose/dataset.csv"
MODEL_DIR = "pose/models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

TARGET_COLUMNS = ["chest", "waist", "hip", "shoulder", "sleeve", "inseam"]

# =========================
# LOAD DATA
# =========================
def load_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    if len(df) < 50:
        raise ValueError("Dataset too small. Collect at least 50+ samples.")

    logger.info(f"Dataset loaded: {df.shape}")

    X = df.drop(columns=TARGET_COLUMNS)
    y = df[TARGET_COLUMNS]

    return X, y


# =========================
# PREPROCESSING
# =========================
def preprocess(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, SCALER_PATH)
    logger.info("Scaler saved.")

    return X_scaled, scaler


# =========================
# TRAIN MODEL
# =========================
def train_model(X, y):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)
    return model


# =========================
# EVALUATE
# =========================
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    logger.info(f"MAE: {mae:.2f} cm")
    logger.info(f"R2 Score: {r2:.3f}")

    return mae, r2


# =========================
# MAIN TRAIN PIPELINE
# =========================
def main():
    X, y = load_data()

    X_scaled, scaler = preprocess(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)

    evaluate(model, X_test, y_test)

    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved: {MODEL_PATH}")


# =========================
# PREDICTION FUNCTION
# =========================
def predict(features_array):
    """
    Input: fused_features (same as dataset row features)
    Output: predicted measurements
    """

    if not os.path.exists(MODEL_PATH):
        raise ValueError("Model not trained yet.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    features_array = np.array(features_array).reshape(1, -1)
    features_scaled = scaler.transform(features_array)

    preds = model.predict(features_scaled)[0]

    result = {
        "chest": round(float(preds[0]), 1),
        "waist": round(float(preds[1]), 1),
        "hip": round(float(preds[2]), 1),
        "shoulder": round(float(preds[3]), 1),
        "sleeve": round(float(preds[4]), 1),
        "inseam": round(float(preds[5]), 1),
    }

    return result


if __name__ == "__main__":
    main()