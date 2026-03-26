import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def load_model(model_name):

    model = joblib.load(os.path.join(MODEL_DIR, f"{model_name}.pkl"))

    if model_name == "krr_model":
        scaler = joblib.load(os.path.join(MODEL_DIR, "krr_scaler.pkl"))
        encoder = joblib.load(os.path.join(MODEL_DIR, "krr_encoder.pkl"))
    else:
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        encoder = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))
        pca = None

    return model, scaler, encoder, pca

import joblib
from keras.models import load_model

def load_ann_best():
    from keras.models import load_model
    import joblib

    model = load_model("models/ann_best.keras")
    scaler = joblib.load("models/ann_scaler_best.pkl")
    columns = joblib.load("models/ann_columns_best.pkl")
    encoders = joblib.load("models/ann_encoders_best.pkl")

    return model, scaler, columns, encoders

def load_hybrid_model():
    import joblib
    from keras.models import load_model

    # ANN
    ann = load_model("models/ann_best.keras")
    ann_scaler = joblib.load("models/ann_scaler_best.pkl")
    ann_columns = joblib.load("models/ann_columns_best.pkl")
    ann_encoders = joblib.load("models/ann_encoders_best.pkl")

    # RF
    rf = joblib.load("models/rf_model.pkl")
    rf_scaler = joblib.load("models/scaler.pkl")
    rf_encoder = joblib.load("models/encoder.pkl")

    # META MODEL (STACKING)
    meta = joblib.load("models/hybrid_meta.pkl")

    return ann, rf, meta, ann_scaler, ann_columns, ann_encoders, rf_scaler, rf_encoder