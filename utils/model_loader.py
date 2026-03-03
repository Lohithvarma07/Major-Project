import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def load_model(model_name):

    model = joblib.load(os.path.join(MODEL_DIR, f"{model_name}.pkl"))

    if model_name == "krr_model":
        scaler = joblib.load(os.path.join(MODEL_DIR, "krr_scaler.pkl"))
        encoder = joblib.load(os.path.join(MODEL_DIR, "krr_encoder.pkl"))
        pca = joblib.load(os.path.join(MODEL_DIR, "krr_pca.pkl"))
    else:
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        encoder = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))
        pca = None

    return model, scaler, encoder, pca