import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(
    "Perovskite_database_content_all_data.csv",
    low_memory=False
)

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

y = np.log1p(df[target].values)

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_cat = encoder.fit_transform(df[categorical_cols])

X = np.concatenate([encoded_cat, df[numeric_cols].values], axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("XGBoost R2:", r2_score(y_test, y_pred))
print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

joblib.dump(model, "models/xgb_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoder, "models/encoder.pkl")

print("XGBoost model saved.")