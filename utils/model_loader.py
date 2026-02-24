import joblib

def load_rf():
    model = joblib.load("models/rf_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/encoder.pkl")
    return model, scaler, encoder