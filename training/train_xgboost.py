import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

encoder = joblib.load("models/encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

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

df = df.dropna(subset=[target])
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[categorical_cols] = df[categorical_cols].fillna("Unknown")
df = df[df[target] < 5000]

y = np.log1p(df[target].values)

encoded_cat = encoder.transform(df[categorical_cols])
X = np.concatenate([encoded_cat, df[numeric_cols].values], axis=1)
X = scaler.transform(X)

print("Feature shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("XGB R2:", r2_score(y_test, y_pred))
print("XGB RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

joblib.dump(model, "models/xgb_model.pkl")
print(" XGB model saved.")