import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

df = pd.read_csv(
    os.path.join(BASE_DIR, "Perovskite_database_content_all_data.csv"),
    low_memory=False
)

# ==========================
# 🔥 NUMERIC ONLY (NO CATEGORICAL)
# ==========================
numeric_cols = [
    "Perovskite_thickness",
    "Perovskite_band_gap",
    "Stability_temperature_range",
    "Stability_relative_humidity_range",
    "Stability_light_intensity",
    "Stability_time_total_exposure"
]

target = "Stability_PCE_T80"

df = df[numeric_cols + [target]]

# ==========================
# CLEAN
# ==========================
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df = df.dropna(subset=[target])

# ==========================
# 🔥 FEATURE ENGINEERING (IMPORTANT)
# ==========================
# FEATURE ENGINEERING
df["interaction"] = df["Perovskite_band_gap"] * df["Perovskite_thickness"]

df["log_thickness"] = np.log1p(df["Perovskite_thickness"])  # ✅ ADD THIS

df["env_effect"] = (
    df["Stability_temperature_range"] *
    df["Stability_relative_humidity_range"]
)

numeric_cols.extend(["interaction", "log_thickness", "env_effect"])

# ==========================
# FINAL FEATURES
# ==========================
X = df[numeric_cols].values
X = np.nan_to_num(X)

# ==========================
# SCALE ONLY (NO ENCODER)
# ==========================
scaler = StandardScaler()
scaler.fit(X)

# ==========================
# SAVE ONLY SCALER
# ==========================
joblib.dump(scaler, os.path.join(MODEL_DIR, "krr_scaler.pkl"))

print("✅ KRR preprocessing fixed (numeric only)")