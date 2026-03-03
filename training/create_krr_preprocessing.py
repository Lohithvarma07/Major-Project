import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

df = pd.read_csv(os.path.join(BASE_DIR, "Perovskite_database_content_all_data.csv"), low_memory=False)

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
df = df.dropna(subset=[target])

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

df[categorical_cols] = df[categorical_cols].fillna("Unknown")

df = df[df[target] < 5000]

# 🔥 For KRR we make dense encoding
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_cat = encoder.fit_transform(df[categorical_cols])

X = np.concatenate([encoded_cat, df[numeric_cols].values], axis=1)

# 🔥 KRR needs scaling
scaler = StandardScaler()
scaler.fit(X)

joblib.dump(encoder, os.path.join(MODEL_DIR, "krr_encoder.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "krr_scaler.pkl"))

print("KRR preprocessing created.")