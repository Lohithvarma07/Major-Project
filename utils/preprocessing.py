import pandas as pd
import numpy as np

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

def prepare_input(user_dict):
    df = pd.DataFrame([user_dict])

    categorical_df = df[categorical_cols]
    numeric_df = df[numeric_cols]

    return categorical_df, numeric_df