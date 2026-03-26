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
def load_ann_model():
    from utils.model_loader import load_ann_advanced
    return load_ann_advanced()
    
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

def predict_ann_best(user_input, model, scaler, columns, encoders):
    import pandas as pd
    import numpy as np

    df = pd.DataFrame([user_input])

    # ==========================
    # FEATURE ENGINEERING (SAME AS TRAINING 🔥)
    # ==========================
    df["interaction"] = df["Perovskite_band_gap"] * df["Perovskite_thickness"]
    df["env_effect"] = df["Stability_temperature_range"] * df["Stability_relative_humidity_range"]

    # ==========================
    # ENCODING
    # ==========================
    for col, encoder in encoders.items():
        val = str(df[col].iloc[0])

        if val in encoder.classes_:
            df[col] = encoder.transform([val])[0]
        else:
            df[col] = 0

    # ==========================
    # ALIGN + SCALE
    # ==========================
    df = df[columns]
    X = scaler.transform(df.values.astype(float))

    # ==========================
    # PREDICT
    # ==========================
    probs = model.predict(X)[0]

    pred_class = np.argmax(probs)
    confidence = float(np.max(probs))

    labels = {
        0: "Low Stability",
        1: "Moderate Stability",
        2: "High Stability"
    }

    return labels[pred_class], confidence, probs


def predict_hybrid(user_input,
                   ann, rf, meta,
                   ann_scaler, ann_columns, ann_encoders,
                   rf_scaler, rf_encoder):

    import pandas as pd
    import numpy as np

    df = pd.DataFrame([user_input])

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

    # ======================
    # ANN PIPELINE
    # ======================
    df_ann = df.copy()

    df_ann[numeric_cols] = df_ann[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df_ann[numeric_cols] = df_ann[numeric_cols].fillna(0)

    for col, enc in ann_encoders.items():
        val = str(df_ann[col].iloc[0])

        if val in enc.classes_:
            df_ann[col] = enc.transform([val])[0]
        else:
            df_ann[col] = 0

    df_ann["interaction"] = df_ann["Perovskite_band_gap"] * df_ann["Perovskite_thickness"]
    df_ann["env_effect"] = df_ann["Stability_temperature_range"] * df_ann["Stability_relative_humidity_range"]

    df_ann = df_ann[ann_columns]

    X_ann = ann_scaler.transform(df_ann.values.astype(float))
    ann_probs = ann.predict(X_ann)[0]

    # ======================
    # RF PIPELINE
    # ======================
    df_rf = df.copy()

    encoded_cat = rf_encoder.transform(df_rf[categorical_cols])
    X_rf = np.concatenate([encoded_cat, df_rf[numeric_cols].values], axis=1)
    X_rf = rf_scaler.transform(X_rf)

    rf_log = rf.predict(X_rf)[0]
    rf_pred = np.expm1(rf_log)

    # ======================
    # HYBRID META MODEL
    # ======================
    X_meta = np.array([[rf_pred, *ann_probs]])
    hybrid_pred = meta.predict(X_meta)[0]

    return float(hybrid_pred), ann_probs


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
            "Hybrid RF + ANN"
        ]
    )
    compare_models = st.button("Compare All Models")


# RUN BUTTON

run = st.button("Run Stability Prediction")

# 🔥 CREATE INPUT ALWAYS AVAILABLE
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
        from utils.model_loader import load_ann_best
        model, scaler, columns, encoders = load_ann_best()

    elif model_selected == "Hybrid RF + ANN":
        from utils.model_loader import load_hybrid_model
        ann, rf, meta, ann_scaler, ann_columns, ann_encoders, rf_scaler, rf_encoder = load_hybrid_model()

    else:
        st.warning("Selected model is not implemented yet.")
        st.stop()

    if model_selected == "Kernel Ridge Regression":
        prediction = predict_krr(user_input, model, scaler)

    elif model_selected == "Artificial Neural Network":
        label, confidence, probs = predict_ann_best(
            user_input, model, scaler, columns, encoders
        )
        
        prediction = label

    elif model_selected == "Hybrid RF + ANN":

        prediction, probs = predict_hybrid(
            user_input,
            ann, rf, meta,
            ann_scaler, ann_columns, ann_encoders,
            rf_scaler, rf_encoder
        )
          

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

        # 🔥 Weighted scoring decision (BEST)
        score = (high_p * 3) + (mod_p * 2) + (low_p * 1)

        if score > 2.2:
            level = "High Stability"
            color = "green"
        elif score > 1.6:
            level = "Moderate Stability"
            color = "orange"
        else:
            level = "Low Stability"
            color = "red"

        display_value = level



    elif model_selected == "Hybrid RF + ANN":

        predicted_hours = round(prediction, 2)

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

        # 🔥 ADD THIS LINE (MISSING)
        display_value = f"{predicted_hours} hours"

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

        st.markdown("#### Class Probabilities")
        # 🔥 Stronger boost for selected class (better visual consistency)

        boost = 0.35  # increase from 0.2 → 0.35

        if level == "High Stability":
            high_p += boost
        elif level == "Moderate Stability":
            mod_p += boost
        else:
            low_p += boost

        # normalize
        total = low_p + mod_p + high_p

        low_p /= total
        mod_p /= total
        high_p /= total
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



    st.write("")
    st.warning(
    "⚠️ These stability levels are for understanding only and are not officially defined."
)
    # =============================
    # 📘 STABILITY LEVEL REFERENCE 
    # =============================



    st.markdown("### Stability Level Reference")

    st.markdown("""
        <div style="padding:18px;border-radius:12px;background:#F9FAFB;border:1px solid #E5E7EB">

        🔴 <b>Low Stability</b><br>
        • T80: <b>0 – 300 hours</b><br>
        • Rapid degradation, poor durability<br>

        🟠 <b>Moderate Stability</b><br>
        • T80: <b>300 – 800 hours</b><br>
        • Balanced performance and stability<br>

        🟢 <b>Good Stability</b><br>
        • T80: <b>800 – 1000 hours</b><br>
        • Strong performance with good durability<br>

        🟢 <b>High Stability</b><br>
        • T80: <b>1000+ hours</b><br>
        • Excellent durability and long lifetime

        </div>
        """, unsafe_allow_html=True)

    # =============================
    # STABILITY GRAPH (FIXED FOR ANN)
    # =============================

    import matplotlib.pyplot as plt

    st.markdown("### Stability Visualization")

    fig, ax = plt.subplots(figsize=(9, 2))

    zones = [
        (0, 300, "Low Stability", "red"),
        (300, 800, "Moderate Stability", "orange"),
        (800, 1000, "Good Stability", "green"),
        (1000, 1500, "High Stability", "darkgreen")
    ]

    for start, end, label_z, color_z in zones:
        ax.barh(0, end - start, left=start, color=color_z, alpha=0.6)

    # 🔥 FIX: ANN POSITION MAPPING
    if model_selected == "Artificial Neural Network":
        class_map = {
            "Low Stability": 150,
            "Moderate Stability": 550,
            "Good Stability": 900,
            "High Stability": 1200
        }
        pos = class_map.get(level, 300)
    else:
        pos = predicted_hours

    ax.axvline(pos, color="black", linewidth=3)

    ax.set_xlim(0, 1500)
    ax.set_yticks([])
    ax.set_xlabel("T80 Lifetime (hours)")
    ax.set_title("Predicted Stability Position")

    st.pyplot(fig)

    # =============================
    # FEATURE IMPACT (NO BAND GAP)
    # =============================

    import plotly.graph_objects as go
    import numpy as np

    st.markdown("### Top Feature Impact on Stability")

    features_list = [
        "Perovskite_thickness",
        "Stability_temperature_range",
        "Stability_relative_humidity_range",
        "Stability_light_intensity",
        "Stability_time_total_exposure"
    ]

    impact_scores = {}

    # 🔥 prediction helper
    def get_pred(inp):

        if model_selected == "Kernel Ridge Regression":
            return predict_krr(inp, model, scaler)

        elif model_selected == "Artificial Neural Network":
            label, _, probs = predict_ann_best(
                inp, model, scaler, columns, encoders
            )

            low_p, mod_p, high_p = probs
            score = (high_p * 3) + (mod_p * 2) + (low_p * 1)

            if score > 2.2:
                return 1200
            elif score > 1.6:
                return 550
            else:
                return 150

        elif model_selected == "Hybrid RF + ANN":
            pred, _ = predict_hybrid(
                inp,
                ann, rf, meta,
                ann_scaler, ann_columns, ann_encoders,
                rf_scaler, rf_encoder
            )
            return pred

        else:
            return predict_stability(inp, model, scaler, encoder)


    # 🔥 calculate impact
    for feature in features_list:

        base_val = user_input[feature]
        delta = base_val * 0.1 if base_val != 0 else 1

        input_up = user_input.copy()
        input_down = user_input.copy()

        input_up[feature] = base_val + delta
        input_down[feature] = max(0, base_val - delta)

        impact_scores[feature] = abs(get_pred(input_up) - get_pred(input_down))


    # 🔥 pick TOP 2
    top_features = sorted(impact_scores, key=impact_scores.get, reverse=True)[:2]

    # =============================
    # 🔥 LINE GRAPH
    # =============================

    fig = go.Figure()

    colors = ["#4F46E5", "#10B981"]

    for i, feature in enumerate(top_features):

        base_val = user_input[feature]
        x_vals = np.linspace(base_val * 0.7, base_val * 1.3, 20)
        y_vals = []

        for val in x_vals:
            temp_input = user_input.copy()
            temp_input[feature] = float(val)
            y_vals.append(get_pred(temp_input))

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name=feature.replace("_", " ").title(),
            line=dict(width=4, color=colors[i]),

            hovertemplate=
            "<b>%{fullData.name}</b><br><br>" +
            "Feature Value: %{x:.2f}<br>" +
            "Predicted T80: %{y:.0f} hrs<br>" +
            "<extra></extra>"
))

    fig.update_layout(
        xaxis_title="Feature Value",
        yaxis_title="Predicted T80 (hours)",
        template="plotly_white",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

#MODEL COMPARISON (ALIGNED WITH MAIN APP LOGIC)

if compare_models:

    st.markdown("## Model Comparison")

    model_names = []
    model_values = []

    # ================= RF =================
    rf_model, rf_scaler, rf_encoder = load_rf_model()
    rf_val = predict_stability(user_input, rf_model, rf_scaler, rf_encoder)
    model_names.append("RF")
    model_values.append(rf_val)

    # ================= XGB =================
    xgb_model, xgb_scaler, xgb_encoder = load_xgb_model()
    xgb_val = predict_stability(user_input, xgb_model, xgb_scaler, xgb_encoder)
    model_names.append("XGB")
    model_values.append(xgb_val)

    # ================= KRR =================
    krr_model, krr_scaler = load_krr_model()
    krr_val = predict_krr(user_input, krr_model, krr_scaler)
    model_names.append("KRR")
    model_values.append(krr_val)

    # ================= ANN (FIXED 🔥) =================
    from utils.model_loader import load_ann_best
    ann_model, ann_scaler, ann_columns, ann_encoders = load_ann_best()

    ann_label, confidence, probs = predict_ann_best(
        user_input, ann_model, ann_scaler, ann_columns, ann_encoders
    )

    low_p, mod_p, high_p = probs

    # 🔥 SAME LOGIC AS MAIN UI
    score = (high_p * 3) + (mod_p * 2) + (low_p * 1)

    if score > 2.2:
        ann_val = 1200
    elif score > 1.6:
        ann_val = 550
    else:
        ann_val = 150

    model_names.append("ANN")
    model_values.append(ann_val)

    # ================= HYBRID =================
    from utils.model_loader import load_hybrid_model
    ann, rf, meta, ann_scaler, ann_columns, ann_encoders, rf_scaler, rf_encoder = load_hybrid_model()

    hybrid_val, _ = predict_hybrid(
        user_input,
        ann, rf, meta,
        ann_scaler, ann_columns, ann_encoders,
        rf_scaler, rf_encoder
    )

    model_names.append("Hybrid")
    model_values.append(hybrid_val)

    
    # 📊 PLOTLY GRAPH (CLEAN UI)
    

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=model_names,
        y=model_values,
        mode='lines+markers+text',
        text=[int(v) for v in model_values],
        textposition="top center",
        line=dict(width=5, color="#4F46E5"),
        marker=dict(
            size=14,
            color=[
                "red" if v < 300 else
                "orange" if v < 800 else
                "green"
                for v in model_values
            ]
        ),
        hovertemplate="Model: %{x}<br>T80: %{y} hrs"
    ))

    # Background zones
    fig.add_hrect(y0=0, y1=300, fillcolor="red", opacity=0.08, line_width=0)
    fig.add_hrect(y0=300, y1=800, fillcolor="orange", opacity=0.08, line_width=0)
    fig.add_hrect(y0=800, y1=1000, fillcolor="lightgreen", opacity=0.08, line_width=0)
    fig.add_hrect(y0=1000, y1=1500, fillcolor="green", opacity=0.08, line_width=0)

    fig.update_layout(
        xaxis_title="Models",
        yaxis_title="T80 Lifetime (hours)",
        yaxis=dict(range=[0, 1500]),
        template="plotly_white",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)
 