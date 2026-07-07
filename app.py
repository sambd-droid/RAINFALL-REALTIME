import os
from datetime import date, timedelta

import joblib
import pandas as pd
import requests
import streamlit as st

# -------------------------------------------------
# Configuration
# -------------------------------------------------
MODEL_PATH = "rainfall_randomforest_80_20.pkl"

# Default location: Dhaka, Bangladesh
LATITUDE = 23.8103
LONGITUDE = 90.4125
TIMEZONE = "Asia/Dhaka"

st.set_page_config(
    page_title="Rainfall Prediction",
    page_icon="🌧️",
    layout="centered"
)

# -------------------------------------------------
# Load saved Random Forest model
# -------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' was not found. "
            "Upload it to the same Streamlit project folder."
        )

    saved = joblib.load(MODEL_PATH)

    return (
        saved["model"],
        saved["feature_names"],
        saved["label_encoder"]
    )

model, feature_names, le = load_model()


def get_class_name(class_id):
    """Convert an encoded predicted class to its original rainfall class name."""
    return str(le.inverse_transform([int(class_id)])[0])


def fetch_weather_features(selected_date):
    """
    Download weather data for one date from Open-Meteo,
    then calculate the 8 feature values required by the model.
    """
    date_text = selected_date.strftime("%Y-%m-%d")

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={date_text}"
        f"&end_date={date_text}"
        "&daily=temperature_2m_max"
        "&hourly=dew_point_2m,relative_humidity_2m,"
        "pressure_msl,visibility,wind_speed_10m"
        f"&timezone={TIMEZONE}"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    hourly_df = pd.DataFrame(data["hourly"])
    daily_df = pd.DataFrame(data["daily"])

    if hourly_df.empty or daily_df.empty:
        raise ValueError("Weather data is not available for this date.")

    # These names must match the 8 features used in model training.
    weather_features = {
        "temp high": float(daily_df["temperature_2m_max"].iloc[0]),
        "DP high": float(hourly_df["dew_point_2m"].max()),
        "month": int(selected_date.month),
        "day": int(selected_date.day),
        "humidity avg": float(hourly_df["relative_humidity_2m"].mean()),
        "SLP avg": float(hourly_df["pressure_msl"].mean()),
        "visibility avg": float(hourly_df["visibility"].mean()),
        "wind avg": float(hourly_df["wind_speed_10m"].mean())
    }

    return weather_features


# -------------------------------------------------
# App interface
# -------------------------------------------------
st.title("🌧️ Localized Rainfall Prediction")
st.write(
    "Choose a date. The app collects weather values automatically "
    "and predicts the rainfall class."
)

st.caption("Location: Dhaka, Bangladesh")

# Open-Meteo forecast availability is limited.
min_date = date.today()
max_date = date.today() + timedelta(days=16)

selected_date = st.date_input(
    "Choose Prediction Date",
    value=date.today(),
    min_value=min_date,
    max_value=max_date
)

if st.button("Get Rainfall Prediction", type="primary"):
    try:
        with st.spinner("Collecting weather data and predicting rainfall..."):
            weather_values = fetch_weather_features(selected_date)

            # Use the original feature order saved with the model.
            input_df = pd.DataFrame([weather_values])[feature_names]

            predicted_id = model.predict(input_df)[0]
            predicted_class = get_class_name(predicted_id)

        st.success(f"Predicted Rainfall Class: {predicted_class}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            confidence = probabilities.max() * 100

            st.info(f"Confidence: {confidence:.2f}%")

            probability_df = pd.DataFrame({
                "Rainfall Class": [get_class_name(class_id) for class_id in model.classes_],
                "Probability (%)": probabilities * 100
            })

            st.subheader("Class Probabilities")
            st.dataframe(probability_df, use_container_width=True)
            st.bar_chart(probability_df.set_index("Rainfall Class"))

        st.subheader("Weather Values Used")
        used_values = pd.DataFrame.from_dict(
            weather_values,
            orient="index",
            columns=["Value"]
        )
        st.dataframe(used_values, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.warning(
            "Please choose a date within the available forecast range. "
            "For past dates, use a historical weather API."
        )

st.caption("Model: Random Forest | Weather source: Open-Meteo")
