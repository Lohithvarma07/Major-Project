import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("Perovskite_database_content_all_data.csv", low_memory=False)

# ==========================
# 🔥 USE STRONG DOMAIN FEATURES ONLY
# ==========================
features = [
    "Perovskite_thickness",
    "Perovskite_band_gap",
    "Stability_temperature_range",
    "Stability_relative_humidity_range",
    "Stability_light_intensity",
    "Stability_time_total_exposure"
]

target = "Stability_PCE_T80"

df = df[features + [target]]

# ==========================
# CLEAN
# ==========================
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df[features] = df[features].fillna(df[features].median())
df = df.dropna(subset=[target])

# Remove noise
df = df[df[target] < df[target].quantile(0.99)]

print("Dataset size:", df.shape)

# ==========================
# 🔥 FEATURE ENGINEERING
# ==========================
df["interaction"] = df["Perovskite_band_gap"] * df["Perovskite_thickness"]
df["log_thickness"] = np.log1p(df["Perovskite_thickness"])
df["env_effect"] = (
    df["Stability_temperature_range"] *
    df["Stability_relative_humidity_range"]
)

features.extend([
    "interaction",
    "log_thickness",
    "env_effect"
])

# ==========================
# FEATURES / TARGET
# ==========================
X = df[features].values
X = np.nan_to_num(X)

y = np.log1p(df[target].values)

# ==========================
# SCALE
# ==========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==========================
# SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# 🔥 KRR TUNING
# ==========================
param_grid = {
    "alpha": [0.01, 0.1, 1],
    "gamma": [0.1, 0.5, 1],
    "kernel": ["laplacian", "rbf"]
}

grid = GridSearchCV(KernelRidge(), param_grid, cv=5, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)

model = grid.best_estimator_

# ==========================
# PREDICT
# ==========================
y_pred = model.predict(X_test)

y_pred = np.expm1(y_pred)
y_test_actual = np.expm1(y_test)

# ==========================
# METRICS
# ==========================
r2 = r2_score(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))



print("\n🔥 FINAL STABLE KRR")
print("R2:", round(r2, 4))
print("RMSE:", round(rmse, 4))
print("Best Params:", grid.best_params_)

# ==========================
# 🔥 SAVE MODEL (ADD THIS)
# ==========================
import joblib
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "krr_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "krr_scaler.pkl"))

print("✅ KRR model saved")