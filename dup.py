import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =============================
# PAGE CONFIG
# =============================

st.set_page_config(
    page_title="Perovskite Stability Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================
# LOAD DATASET
# =============================

DATA_PATH = "Perovskite_database_content_all_data.csv"

@st.cache_data
def load_dataset(path):
    return pd.read_csv(path, low_memory=False)

df = load_dataset(DATA_PATH)

# =============================
# HELPER FUNCTIONS
# =============================

def unique_vals(col):
    return sorted(df[col].dropna().astype(str).unique())

def num_range(col):
    series = pd.to_numeric(df[col], errors="coerce")
    valid = series.dropna()

    if valid.empty:
        st.warning(f"No valid numeric values found in column: {col}")
        return 0.0, 1.0, 0.5

    return float(valid.min()), float(valid.max()), float(valid.median())

# =============================
# LOAD RANDOM FOREST MODEL
# =============================

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

# =============================
# PREDICTION FUNCTION
# =============================
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
# =============================
# CSS
# =============================

st.markdown("""
<style>
.block-container { padding-top: 0rem !important; }

.app-header {
    background-color:#0B2C5D;
    padding:28px;
    border-radius:12px;
    color:white;
    text-align:center;
    margin-bottom:30px;
}

.score-bar {
    border:3px solid black;
    border-radius:50px;
    padding:18px;
    text-align:center;
    font-size:22px;
    font-weight:600;
    margin-top:30px;
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

# =============================
# TITLE
# =============================

st.markdown("""
<div class="app-header">
    <h1>Predicting Perovskite Solar Cell Stability</h1>
    <h1>With Machine Learning</h1>
</div>
""", unsafe_allow_html=True)

st.success(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# =============================
# MAIN UI
# =============================

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
    thickness = st.slider("Perovskite Thickness (nm)", t_min, t_max, t_med)

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

    st.info(f"**Current Model:** {model_selected}")

# =============================
# RUN BUTTON
# =============================

run = st.button("Run Stability Prediction")

# =============================
# OUTPUT
# =============================

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

    # =============================
    # MODEL SELECTION
    # =============================

    if model_selected == "Random Forest":
        model, scaler, encoder = load_rf_model()

    elif model_selected == "XGBoost":
        model, scaler, encoder = load_xgb_model()

    else:
        st.warning("Selected model is not implemented yet.")
        st.stop()

    prediction = predict_stability(user_input, model, scaler, encoder)
    predicted_hours = round(prediction, 2)

    # =============================
    # STABILITY LEVEL CLASSIFICATION
    # =============================

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



    st.markdown("""
        <div class="score-bar">
            Predicted Stability Result
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Predicted T80 Lifetime</div>
                <div class="summary-value">{predicted_hours} hours</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Stability Level</div>
                <div class="summary-value">{level}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
