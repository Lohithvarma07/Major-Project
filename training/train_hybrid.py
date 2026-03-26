import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# ==========================
# LOAD MODELS
# ==========================
ann = load_model("models/ann_best.keras")
rf = joblib.load("models/rf_model.pkl")

ann_scaler = joblib.load("models/ann_scaler_best.pkl")
ann_columns = joblib.load("models/ann_columns_best.pkl")
ann_encoders = joblib.load("models/ann_encoders_best.pkl")

rf_scaler = joblib.load("models/scaler.pkl")
rf_encoder = joblib.load("models/encoder.pkl")

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

df = df[categorical_cols + numeric_cols + [target]].copy()

# ==========================
# CLEAN DATA
# ==========================
df = df.dropna(subset=[target])
df = df[(df[target] > 10) & (df[target] < 5000)]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# ==========================
# -------- ANN PIPELINE -------
# ==========================
df_ann = df.copy()

# FIX numeric (CRITICAL 🔥)
df_ann[numeric_cols] = df_ann[numeric_cols].apply(pd.to_numeric, errors="coerce")
df_ann[numeric_cols] = df_ann[numeric_cols].fillna(df_ann[numeric_cols].median())

# encode categorical
for col, enc in ann_encoders.items():
    df_ann[col] = df_ann[col].astype(str)
    df_ann[col] = df_ann[col].map(lambda x: x if x in enc.classes_ else enc.classes_[0])
    df_ann[col] = enc.transform(df_ann[col])

# feature engineering
df_ann["interaction"] = df_ann["Perovskite_band_gap"] * df_ann["Perovskite_thickness"]
df_ann["env_effect"] = df_ann["Stability_temperature_range"] * df_ann["Stability_relative_humidity_range"]

# align columns
df_ann = df_ann[ann_columns]

X_ann = ann_scaler.transform(df_ann.values.astype(float))
ann_probs = ann.predict(X_ann)

# ==========================
# -------- RF PIPELINE -------
# ==========================
encoded_cat = rf_encoder.transform(df[categorical_cols])

X_rf = np.concatenate([encoded_cat, df[numeric_cols].values], axis=1)
X_rf = rf_scaler.transform(X_rf)

rf_log_pred = rf.predict(X_rf)

# 🔥 CRITICAL FIX (log → actual)
rf_pred = np.expm1(rf_log_pred)

# ==========================
# TARGET
# ==========================
y_true = df[target].values

# ==========================
# STACKING (HYBRID)
# ==========================
X_meta = np.column_stack([
    rf_pred,
    ann_probs
])

meta = Ridge(alpha=1.0)
meta.fit(X_meta, y_true)

hybrid_pred = meta.predict(X_meta)

# ==========================
# METRICS
# ==========================
r2 = r2_score(y_true, hybrid_pred)

print("\n==============================")
print(f"🔥 FINAL HYBRID R2 : {r2:.4f}")
print("==============================\n")

# ==========================
# SAVE HYBRID MODEL
# ==========================
joblib.dump(meta, "models/hybrid_meta.pkl")

print("✅ HYBRID META MODEL SAVED")