# predict.py
import numpy as np
import joblib

def predict_units_sold(input_data: dict):
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    # Order: ['price', 'promotion', 'holiday', 'weather_index', 'day_of_week', 'month']
    features = np.array([
        input_data['price'],
        input_data['promotion'],
        input_data['holiday'],
        input_data['weather_index'],
        input_data['day_of_week'],
        input_data['month']
    ]).reshape(1, -1)

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    return round(prediction)
