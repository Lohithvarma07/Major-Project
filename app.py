import streamlit as st
import pandas as pd
import numpy as np


# PAGE CONFIG

st.set_page_config(
    page_title="Perovskite Stability Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# LOAD DATASET (INTERNAL)

DATA_PATH = "Perovskite_database_content_all_data.csv"

@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)

df = load_dataset(DATA_PATH)


# HELPER FUNCTIONS (DATA-DRIVEN UI)

def unique_vals(col):
    return sorted(df[col].dropna().astype(str).unique())

def num_range(col):
    series = pd.to_numeric(df[col], errors="coerce")
    valid = series.dropna()

    if valid.empty:
        st.warning(f"No valid numeric values found in column: {col}")
        return 0.0, 1.0, 0.5

    return float(valid.min()), float(valid.max()), float(valid.median())



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



# TITLE

st.markdown("""
<div class="app-header">
    <h1>Predicting Perovskite Solar Cell Stability</h1>
    <h1>With Deep Learning</h1>
    
</div>
""", unsafe_allow_html=True)

st.success(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")



# MAIN UI (2 COLUMNS)

left_col, right_col = st.columns([2, 1])

# ---------------- LEFT: PARAMETERS ----------------
with left_col:
    st.markdown("### Parameters Selection")

    perovskite_comp = st.selectbox(
        "Perovskite Composition (short form)",
        unique_vals("Perovskite_composition_short_form")
    )

    etl = st.selectbox(
        "Electron Transport Layer (ETL)",
        unique_vals("ETL_stack_sequence")
    )

    htl = st.selectbox(
        "Hole Transport Layer (HTL)",
        unique_vals("HTL_stack_sequence")
    )

    t_min, t_max, t_med = num_range("Perovskite_thickness")
    thickness = st.slider(
        "Perovskite Thickness (nm)",
        t_min, t_max, t_med
    )

    bg_min, bg_max, bg_med = num_range("Perovskite_band_gap")
    bandgap = st.slider(
        "Perovskite Band Gap (eV)",
        bg_min, bg_max, bg_med
    )

    encapsulation = st.selectbox(
        "Encapsulation",
        unique_vals("Encapsulation")
    )

    temp_min, temp_max, temp_med = num_range("Stability_temperature_range")
    temperature = st.slider(
        "Temperature (°C)",
        temp_min, temp_max, temp_med
    )

    hum_min, hum_max, hum_med = num_range("Stability_relative_humidity_range")
    humidity = st.slider(
        "Relative Humidity (%)",
        hum_min, hum_max, hum_med
    )

    light_min, light_max, light_med = num_range("Stability_light_intensity")
    light_intensity = st.slider(
        "Light Intensity (W/m²)",
        light_min, light_max, light_med
    )

    time_min, time_max, time_med = num_range("Stability_time_total_exposure")
    exposure_time = st.slider(
        "Total Exposure Time (hours)",
        time_min, time_max, time_med
    )


# ---------------- RIGHT: MODEL INFO ----------------
with right_col:
    st.markdown("### Model Selection")

    model_selected = st.radio(
        "Prediction Model",
        [
            "Random Forest",
            "Kernel Ridge Regression",
            "XGBoost",
            "CNN (Degradation Curve)",
            "Hybrid ML + DL"
        ]
    )

    st.info(f"**Current Model:** {model_selected}")


# RUN BUTTON

run = st.button("Run Stability Prediction")


# PLACEHOLDER PIPELINE

def preprocess_inputs(user_inputs, dataset):
    pass

def load_model(name):
    pass

def predict_stability(x):
    pass


# OUTPUT

if run:
    st.markdown("""
    <div class="score-bar">
        Predicted Stability Score
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="summary-grid">
        <div class="summary-card">
            <div class="summary-title">Predicted Stability</div>
            <div class="summary-value">—</div>
        </div>
        <div class="summary-card">
            <div class="summary-title">Estimated Lifetime (T80)</div>
            <div class="summary-value">—</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
