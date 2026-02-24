import numpy as np
from utils.preprocessing import prepare_input
from utils.model_loader import load_rf

def predict_stability(user_dict):

    model, scaler, encoder = load_rf()

    categorical_df, numeric_df = prepare_input(user_dict)

    encoded_cat = encoder.transform(categorical_df)
    combined = np.concatenate([encoded_cat, numeric_df.values], axis=1)

    final_input = scaler.transform(combined)

    prediction = model.predict(final_input)

    return float(prediction[0])