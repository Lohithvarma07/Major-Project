import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(
    "Perovskite_database_content_all_data.csv",
    low_memory=False
)

# ==========================
# SELECT FEATURES
# ==========================

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

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# Remove extreme outliers (optional but recommended)
df = df[df[target] < 5000]

# ==========================
# LOG TRANSFORM TARGET
# ==========================

y = np.log1p(df[target].values)

# ==========================
# ENCODING
# ==========================

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_cat = encoder.fit_transform(df[categorical_cols])

X = np.concatenate([encoded_cat, df[numeric_cols].values], axis=1)

# ==========================
# SCALING
# ==========================

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==========================
# TRAIN / TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# RANDOM FOREST
# ==========================

model = RandomForestRegressor(
    n_estimators=800,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================
# EVALUATION
# ==========================

y_pred = model.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# ==========================
# SAVE MODELS
# ==========================

joblib.dump(model, "models/rf_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoder, "models/encoder.pkl")
print(" Training complete. Models saved.")