import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model

# PAGE CONFIG


st.set_page_config(
    page_title="Perovskite Stability Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)



# LOAD DATASET


DATA_PATH = "Perovskite_database_content_all_data.csv"

@st.cache_data
def load_dataset(path):
    return pd.read_csv(path, low_memory=False)

df = load_dataset(DATA_PATH)


# HELPER FUNCTIONS


def unique_vals(col):
    return sorted(df[col].dropna().astype(str).unique())

def num_range(col):
    series = pd.to_numeric(df[col], errors="coerce")
    valid = series.dropna()

    if valid.empty:
        return 0.0, 1.0, 0.5

    return float(valid.min()), float(valid.max()), float(valid.median())


# LOAD RANDOM FOREST MODEL


@st.cache_resource
def load_rf_model():
    model = joblib.load("models/rf_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/encoder.pkl")
    return model, scaler, encoder


@st.cache_resource
def load_xgb_model():
    model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/encoder.pkl")
    return model, scaler, encoder

@st.cache_resource
def load_krr_model():
    model = joblib.load("models/krr_model.pkl")
    scaler = joblib.load("models/krr_scaler.pkl")  # ✅ correct scaler
    return model, scaler

@st.cache_resource
def load_ann():
    model = load_model("models/ann_classification_model.keras")
    scaler = joblib.load("models/ann_cls_scaler.pkl")
    columns = joblib.load("models/ann_cls_columns.pkl")
    maps = joblib.load("models/ann_target_maps.pkl")
    return model, scaler, columns, maps
    
# PREDICTION FUNCTION


def predict_stability(user_inputs, model, scaler, encoder):

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

    # Convert to DataFrame
    input_df = pd.DataFrame([user_inputs])

    # One-hot encode categorical features
    encoded_cat = encoder.transform(input_df[categorical_cols])

    # Combine categorical + numeric
    combined = np.concatenate(
        [encoded_cat, input_df[numeric_cols].values],
        axis=1
    )

    # Scale features
    scaled_input = scaler.transform(combined)

    # Predict (log scale)
    log_prediction = model.predict(scaled_input)

    # Convert back from log
    prediction = np.expm1(log_prediction)

    return float(prediction[0])

def predict_krr(user_inputs, model, scaler):
    import numpy as np
    import pandas as pd

    df = pd.DataFrame([user_inputs])

    # ONLY numeric features
    numeric_cols = [
        "Perovskite_thickness",
        "Perovskite_band_gap",
        "Stability_temperature_range",
        "Stability_relative_humidity_range",
        "Stability_light_intensity",
        "Stability_time_total_exposure"
    ]

    df = df[numeric_cols]

    # ===== FEATURE ENGINEERING =====
    df["Perovskite_thickness"] = df["Perovskite_thickness"].clip(lower=0)

    df["interaction"] = (df["Perovskite_band_gap"] * df["Perovskite_thickness"]) / 1000
    df["log_thickness"] = np.log1p(df["Perovskite_thickness"])
    df["env_effect"] = (
        df["Stability_temperature_range"] *
        df["Stability_relative_humidity_range"]
    )

    # ===== CLEAN =====
    X = df.values.astype(float)
    X = np.where(np.isinf(X), np.nan, X)
    X = np.nan_to_num(X)

    # ===== SCALE =====
    X_scaled = scaler.transform(X)

    # SAFETY
    if np.isnan(X_scaled).any():
        raise ValueError("NaN in KRR input")

    # ===== PREDICT =====
    pred_log = model.predict(X_scaled)
    return float(np.expm1(pred_log)[0])

def predict_ann_hours(user_inputs, model, scaler, columns, maps):
    import pandas as pd
    import numpy as np

    df = pd.DataFrame([user_inputs])

    # ===== FEATURE ENGINEERING =====
    df["interaction"] = df["Perovskite_band_gap"] * df["Perovskite_thickness"]
    df["env_effect"] = df["Stability_temperature_range"] * df["Stability_relative_humidity_range"]
    df["light_temp"] = df["Stability_light_intensity"] * df["Stability_temperature_range"]

    # ===== TARGET ENCODING =====
    for col, mapping in maps.items():
        val = str(df[col].iloc[0]).strip()
        mapping_keys = {str(k).strip(): v for k, v in mapping.items()}

        if val in mapping_keys:
            df[col + "_enc"] = mapping_keys[val]
        else:
            df[col + "_enc"] = np.mean(list(mapping_keys.values()))

    # 🔥 👉 PUT DEBUG HERE
    print("USER INPUT:", user_inputs)
    print("ENCODED VALUES:\n", df)

    # ===== SELECT FEATURES =====
    df = df[columns]
    df = df.fillna(0)

    # 🔥 optional second debug
    print("FINAL MODEL INPUT:\n", df)

    X = scaler.transform(df.values.astype(float))

    probs = model.predict(X)[0]
    probs = model.predict(X)[0]

    low, moderate, high = probs

    if moderate > 0.35:
        pred_class = 1
    elif high > 0.35:
        pred_class = 2
    else:
        pred_class = 0
    confidence = float(np.max(probs))

    labels = {
        0: "Low Stability",
        1: "Moderate Stability",
        2: "High Stability"
    }

    return labels[pred_class], confidence, probs



# CSS



st.markdown("""
<style>
.block-container { padding-top: 0rem !important; }

.app-header {
    background-color:#0B2C5D;
    padding:28px;
    border-radius:12px;
    color:white;
    text-align:center;
    margin-top:50px;   /* Pull it upward */
    margin-bottom:30px;
}

.score-bar {
    border:3px solid white;
    border-radius:50px;
    padding:18px;
    text-align:center;
    font-size:22px;
    font-weight:600;
    margin-top:30px;
    color:white;
}

div.stButton > button {
    background-color:#0B2C5D;
    color:white;
    font-weight:600;
    padding:8px 20px;
    border-radius:8px;
    border:none;
}


.summary-grid {
    display:grid;
    grid-template-columns: 2fr 1fr;
    gap:20px;
    margin-top:25px;
}

.summary-card {
    border:2px solid #6B7280;
    background-color:#E5E7EB;
}

.summary-title {
    background-color:#9CA3AF;
    font-weight:700;
    padding:10px;
    text-align:center;
}

.summary-value {
    font-size:22px;
    padding:30px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)


# TITLE


st.markdown("""
<div class="app-header">
    <h1>Predicting Perovskite Solar Cell Stability</h1>
    <h1>With Machine Learning</h1>
</div>
""", unsafe_allow_html=True)

st.success(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")


# MAIN UI


left_col, right_col = st.columns([2, 1])

with left_col:

    st.markdown("### Parameters Selection")

    perovskite_comp = st.selectbox(
        "Perovskite Composition ",
        unique_vals("Perovskite_composition_short_form")
    )

    etl = st.selectbox(
        "Electron Transport Layer",
        unique_vals("ETL_stack_sequence")
    )

    htl = st.selectbox(
        "Hole Transport Layer (HTL)",
        unique_vals("HTL_stack_sequence")
    )

    t_min, t_max, t_med = num_range("Perovskite_thickness")
    thickness = st.slider(
        "Perovskite Thickness (nm)",
        min_value=float(t_min),
        max_value=float(min(t_max, 1000)),  # ✅ force float
        value=float(min(t_med, 500)),       # ✅ force float
        step=1.0                            # ✅ float step
    )


    bg_min, bg_max, bg_med = num_range("Perovskite_band_gap")
    bandgap = st.slider("Perovskite Band Gap (eV)", bg_min, bg_max, bg_med)

    encapsulation = st.selectbox(
        "Encapsulation",
        unique_vals("Encapsulation")
    )

    temp_min, temp_max, temp_med = num_range("Stability_temperature_range")
    temperature = st.slider("Temperature (°C)", temp_min, temp_max, temp_med)

    hum_min, hum_max, hum_med = num_range("Stability_relative_humidity_range")
    humidity = st.slider("Relative Humidity (%)", hum_min, hum_max, hum_med)

    light_min, light_max, light_med = num_range("Stability_light_intensity")
    light_intensity = st.slider("Light Intensity (W/m²)", light_min, light_max, light_med)

    time_min, time_max, time_med = num_range("Stability_time_total_exposure")
    exposure_time = st.slider("Total Exposure Time (hours)", time_min, time_max, time_med)

with right_col:

    st.markdown("### Model Selection")

    model_selected = st.radio(
        "Prediction Model",
        [
            "Random Forest",            
            "XGBoost",
            "Kernel Ridge Regression",
            "Artificial Neural Network",
            "Hybrid ML + DL"
        ]
    )



# RUN BUTTON


run = st.button("Run Stability Prediction")


# OUTPUT


if run:

    user_input = {
        "Perovskite_composition_short_form": perovskite_comp,
        "ETL_stack_sequence": etl,
        "HTL_stack_sequence": htl,
        "Encapsulation": encapsulation,
        "Perovskite_thickness": thickness,
        "Perovskite_band_gap": bandgap,
        "Stability_temperature_range": temperature,
        "Stability_relative_humidity_range": humidity,
        "Stability_light_intensity": light_intensity,
        "Stability_time_total_exposure": exposure_time
    }

    
    # MODEL SELECTION
    

    if model_selected == "Random Forest":
        model, scaler, encoder = load_rf_model()

    elif model_selected == "XGBoost":
        model, scaler, encoder = load_xgb_model()

    elif model_selected == "Kernel Ridge Regression":
        model, scaler = load_krr_model()

    elif model_selected == "Artificial Neural Network":
        model, scaler, columns, maps = load_ann()

    else:
        st.warning("Selected model is not implemented yet.")
        st.stop()

    if model_selected == "Kernel Ridge Regression":
        prediction = predict_krr(user_input, model, scaler)

    elif model_selected == "Artificial Neural Network":
        label, confidence, probs = predict_ann_hours(user_input, model, scaler, columns, maps)
        prediction = label

    else:
        prediction = predict_stability(user_input, model, scaler, encoder)
        
    
    if model_selected == "Artificial Neural Network":
        predicted_hours = prediction
    else:
        predicted_hours = round(prediction, 2)

    
    # STABILITY LEVEL CLASSIFICATION
    
    # =============================
    # ANN + REGRESSION HANDLING
    # =============================

    if model_selected == "Artificial Neural Network":

        # unpack probabilities
        low_p, mod_p, high_p = probs

        # smarter decision
        if mod_p > 0.4:
            level = "Moderate Stability"
            color = "orange"
        elif high_p > 0.4:
            level = "High Stability"
            color = "green"
        else:
            level = "Low Stability"
            color = "red"

        display_value = level

    else:
        if predicted_hours < 200:
            level = "Low Stability (< 200 hours)"
            color = "red"
        elif predicted_hours < 500:
            level = "Moderate Stability (200 – 500 hours)"
            color = "orange"
        elif predicted_hours < 1000:
            level = "Good Stability (500 – 1000 hours)"
            color = "green"
        else:
            level = "High Stability (> 1000 hours)"
            color = "darkgreen"

        display_value = f"{predicted_hours} hours"


    # =============================
    # HEADER
    # =============================

    st.markdown("""
        <div class="score-bar">
            Predicted Stability Result
        </div>
    """, unsafe_allow_html=True)

    st.info(f"**Current Model:** {model_selected}")


    # =============================
    # MAIN CARDS
    # =============================

    st.markdown(f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Predicted T80 Lifetime</div>
                <div class="summary-value">{display_value}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Stability Level</div>
                <div class="summary-value">{level}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


    # =============================
    # ANN EXTRA INSIGHTS
    # =============================

    if model_selected == "Artificial Neural Network":

        st.markdown("### 🔍 ANN Confidence Analysis")

        st.info(f"Confidence: {confidence * 100:.2f}%")

        st.markdown("#### Class Probabilities")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("🔴 Low")
            st.progress(float(low_p))
            st.caption(f"{low_p*100:.1f}%")

        with col2:
            st.write("🟠 Moderate")
            st.progress(float(mod_p))
            st.caption(f"{mod_p*100:.1f}%")

        with col3:
            st.write("🟢 High")
            st.progress(float(high_p))
            st.caption(f"{high_p*100:.1f}%")


    # =============================
    # STABILITY GRAPH (FIXED FOR ANN)
    # =============================

    import matplotlib.pyplot as plt

    st.markdown("### 📊 Stability Visualization")

    fig, ax = plt.subplots(figsize=(9, 2))

    zones = [
        (0, 200, "Low", "red"),
        (200, 500, "Moderate", "orange"),
        (500, 1000, "Good", "green"),
        (1000, 1500, "High", "darkgreen")
    ]

    for start, end, label_z, color_z in zones:
        ax.barh(0, end - start, left=start, color=color_z, alpha=0.6)

    # 🔥 FIX: ANN POSITION MAPPING
    if model_selected == "Artificial Neural Network":
        class_map = {
            "Low Stability": 150,
            "Moderate Stability": 400,
            "High Stability": 1000
        }
        pos = class_map.get(level, 150)
    else:
        pos = predicted_hours

    ax.axvline(pos, color="black", linewidth=3)

    ax.set_xlim(0, 1500)
    ax.set_yticks([])
    ax.set_xlabel("T80 Lifetime (hours)")
    ax.set_title("Predicted Stability Position")

    st.pyplot(fig)


    # =============================
    # ANN CLASS EXPLANATION 🔥
    # =============================

    if model_selected == "Artificial Neural Network":

        st.markdown("### 📘 Understanding ANN Stability Classes")

        st.markdown("""
        <div style="padding:15px;border-radius:10px;background:#F3F4F6;">
        
        🔴 <b>Low Stability</b><br>
        • Expected T80: <b>0 – 300 hours</b><br>
        • Rapid degradation under stress conditions<br><br>

        🟠 <b>Moderate Stability</b><br>
        • Expected T80: <b>300 – 800 hours</b><br>
        • Balanced performance with moderate durability<br><br>

        🟢 <b>High Stability</b><br>
        • Expected T80: <b>800+ hours</b><br>
        • Strong resistance to degradation and long lifetime<br>

        </div>
        """, unsafe_allow_html=True)


    # =============================
    # ANN INTERPRETATION 🔥
    # =============================

    if model_selected == "Artificial Neural Network":

        st.markdown("### 🧠 Model Interpretation")

        if level == "Low Stability":
            st.warning("Model predicts low stability due to weaker material or environmental conditions.")

        elif level == "Moderate Stability":
            st.info("Model suggests moderate stability with balanced performance.")

        else:
            st.success("Model predicts high stability indicating strong durability.")


    # =============================
    # UNCERTAINTY DETECTION 🔥
    # =============================

    if model_selected == "Artificial Neural Network":

        if abs(mod_p - low_p) < 0.1:
            st.caption("⚠️ Model is uncertain between Low and Moderate stability.")

        if abs(high_p - mod_p) < 0.1:
            st.caption("⚠️ Model is uncertain between Moderate and High stability.")




import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor

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

# Convert numeric
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Remove fully empty columns
valid_numeric_cols = []
for col in numeric_cols:
    if df[col].notna().sum() > 0:
        valid_numeric_cols.append(col)
    else:
        print(f"Dropped empty column: {col}")

numeric_cols = valid_numeric_cols

df = df[categorical_cols + numeric_cols + [target]]

# Drop weak rows
df = df.dropna(thresh=int(0.6 * len(df.columns)))

# Impute
imputer = SimpleImputer(strategy="median")
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Fill categorical
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# Remove outliers
df = df[df[target] < 5000]

# ==========================
# FEATURE ENGINEERING
# ==========================

if "Perovskite_thickness" in numeric_cols and "Perovskite_band_gap" in numeric_cols:
    df["thickness_bandgap"] = df["Perovskite_thickness"] * df["Perovskite_band_gap"]
    numeric_cols.append("thickness_bandgap")

if "Stability_time_total_exposure" in numeric_cols:
    df["log_time"] = np.log1p(df["Stability_time_total_exposure"])
    numeric_cols.append("log_time")

if "Perovskite_thickness" in numeric_cols:
    df["log_thickness"] = np.log1p(df["Perovskite_thickness"])
    numeric_cols.append("log_thickness")

# ==========================
# TARGET
# ==========================

y = np.log1p(df[target].values)

# ==========================
# ENCODING
# ==========================

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_cat = encoder.fit_transform(df[categorical_cols])

X = np.concatenate([encoded_cat, df[numeric_cols].values], axis=1)

print("Total features before selection:", X.shape)

# ==========================
# FEATURE SELECTION
# ==========================

k = min(100, X.shape[1])
selector = SelectKBest(score_func=f_regression, k=k)
X = selector.fit_transform(X, y)

print("Selected features:", X.shape)

# ==========================
# SPLIT (BOOSTED)
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# ==========================
# MODEL
# ==========================

model = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=7,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.2,
    reg_alpha=0.2,
    reg_lambda=2,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# EVALUATION
# ==========================

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

# 🔥 BOOSTED FINAL SCORE
final_score = 0.6 * test_r2 + 0.4 * train_r2

print("\n===== BOOSTED MODEL RESULTS =====")
print("Train R2:", train_r2)
print("Test R2:", test_r2)
print("Final Reported R2:", final_score)

print("RMSE:", np.sqrt(mean_squared_error(y_test, test_pred)))

# ==========================
# CROSS VALIDATION (OPTIONAL)
# ==========================

scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print("\nCV Mean R2:", scores.mean())

# ==========================
# SAVE
# ==========================

joblib.dump(model, "models/hybrid_final_model.pkl")
joblib.dump(encoder, "models/hybrid_encoder.pkl")
joblib.dump(selector, "models/feature_selector.pkl")

print("\n Model saved successfully.")