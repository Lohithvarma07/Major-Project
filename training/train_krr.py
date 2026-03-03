import os
import pandas as pd
import numpy as np
import joblib
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA


# PATH SETUP


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "Perovskite_database_content_all_data.csv")


# LOAD DATA


df = pd.read_csv(DATA_PATH, low_memory=False)

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


# ENCODING


encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_cat = encoder.fit_transform(df[categorical_cols])

X = np.concatenate([encoded_cat, df[numeric_cols].values], axis=1)


# SCALING


scaler = StandardScaler()
X = scaler.fit_transform(X)

X = np.nan_to_num(X)

print("Before PCA shape:", X.shape)


# PCA (DIMENSION REDUCTION)


pca = PCA(n_components=80)  # try 50–120 if needed
X = pca.fit_transform(X)

print("After PCA shape:", X.shape)

# Save preprocessing
joblib.dump(encoder, os.path.join(MODEL_DIR, "krr_encoder.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "krr_scaler.pkl"))
joblib.dump(pca, os.path.join(MODEL_DIR, "krr_pca.pkl"))


# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# HYPERPARAMETER TUNING


param_grid = {
    "alpha": [0.01, 0.1, 1, 5, 10, 50],
    "kernel": ["linear", "rbf"],
    "gamma": [0.001, 0.01, 0.1]
}

grid = GridSearchCV(
    KernelRidge(),
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Params:", grid.best_params_)


# EVALUATION


y_pred = best_model.predict(X_test)

print("KRR R2:", r2_score(y_test, y_pred))
print("KRR RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

joblib.dump(best_model, os.path.join(MODEL_DIR, "krr_model.pkl"))

print("KRR with PCA saved.")