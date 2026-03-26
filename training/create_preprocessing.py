import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("Perovskite_database_content_all_data.csv", low_memory=False)

categorical_cols = [
    "Perovskite_composition_short_form",
    "ETL_stack_sequence",
    "HTL_stack_sequence",
    "Encapsulation"
]

numeric_cols = [
    "Perovskite_thickness",
    "Perovskite_band_gap",
    "Stability_temperature_range",
    "Stability_relative_humidity_range",
    "Stability_light_intensity",
    "Stability_time_total_exposure"
]

target = "Stability_PCE_T80"

df = df[categorical_cols + numeric_cols + [target]]

# ==========================
# CLEANING
# ==========================

df = df.dropna(subset=[target])

# Ensure numeric
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# 🔥 Robust outlier removal
df = df[df[target] < df[target].quantile(0.98)]

# ==========================
# 🔥 FEATURE ENGINEERING (SAFE + HIGH IMPACT)
# ==========================

df["thickness_bandgap"] = df["Perovskite_thickness"] * df["Perovskite_band_gap"]
df["light_time"] = df["Stability_light_intensity"] * df["Stability_time_total_exposure"]

# Only strong features (avoid noise)
numeric_cols = numeric_cols + [
    "thickness_bandgap",
    "light_time"
]

# ==========================
# ENCODING
# ==========================

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_cat = encoder.fit_transform(df[categorical_cols])

# ==========================
# FEATURE MATRIX
# ==========================

X = np.concatenate([encoded_cat, df[numeric_cols].values], axis=1)

# ==========================
# SCALING (IMPORTANT FOR KRR & ANN)
# ==========================

scaler = StandardScaler()
scaler.fit(X)

# ==========================
# SAVE
# ==========================

joblib.dump(encoder, "models/encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Universal preprocessing ready for RF, XGB, KRR, ANN, Hybrid")