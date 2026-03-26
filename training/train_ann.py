import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("Perovskite_database_content_all_data.csv", low_memory=False)

target = "Stability_PCE_T80"

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

df = df[categorical_cols + numeric_cols + [target]].dropna()

# ==========================
# CLEANING
# ==========================
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# ==========================
# ENCODING
# ==========================
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

joblib.dump(encoders, "models/ann_encoders_best.pkl")

# ==========================
# FEATURE ENGINEERING 🔥
# ==========================
df["interaction"] = df["Perovskite_band_gap"] * df["Perovskite_thickness"]
df["env_effect"] = df["Stability_temperature_range"] * df["Stability_relative_humidity_range"]

feature_cols = categorical_cols + numeric_cols + ["interaction", "env_effect"]

X = df[feature_cols].values

# ==========================
# CLASSIFICATION TARGET
# ==========================
df["T80_class"] = pd.cut(
    df[target],
    bins=[0, 500, 2000, 10000],
    labels=[0, 1, 2]
)

y_class = df["T80_class"].astype(int).values
y = to_categorical(y_class)

# ==========================
# SCALING
# ==========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, "models/ann_scaler_best.pkl")
joblib.dump(feature_cols, "models/ann_columns_best.pkl")

# ==========================
# SPLIT
# ==========================
X_train, X_test, y_train, y_test, y_class_train, y_class_test = train_test_split(
    X, y, y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

# ==========================
# ANN MODEL (SIMPLE = BETTER 🔥)
# ==========================
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# TRAIN
# ==========================
model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ==========================
# EVALUATION
# ==========================
y_pred = np.argmax(model.predict(X_test), axis=1)

acc = accuracy_score(y_class_test, y_pred)

print("\n==============================")
print(f"🔥 FINAL ANN Accuracy: {acc:.4f}")
print("==============================\n")

print("Classification Report:\n", classification_report(y_class_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_class_test, y_pred))

# ==========================
# SAVE MODEL
# ==========================
model.save("models/ann_best.keras")

print("✅ BEST ANN MODEL SAVED")