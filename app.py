import os
from datetime import date, timedelta

import joblib
import pandas as pd
import requests
import streamlit as st

MODEL_PATH = "rainfall_randomforest_80_20.joblib"

LATITUDE = 23.8103
LONGITUDE = 90.4125
TIMEZONE = "Asia/Dhaka"

st.set_page_config(
    page_title="Localized Rainfall Prediction",
    page_icon="🌧️",
    layout="wide"
)

# -----------------------------
# White card design CSS
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: #eef3f7;
    }

    .block-container {
        max-width: 1100px;
        margin-top: 70px;
        padding: 55px 50px;
        background: white;
        border-radius: 18px;
        box-shadow: 0px 10px 35px rgba(0,0,0,0.08);
    }

    h1 {
        text-align: center;
        color: #5f6f82;
        font-size: 46px !important;
        font-weight: 800 !important;
    }

    .subtitle {
        text-align: center;
        color: #5f6f82;
        font-size: 22px;
        margin-bottom: 35px;
    }

    .location-text {
        text-align: center;
        color: #5f6f82;
        font-size: 18px;
        margin-top: 25px;
        margin-bottom: 25px;
    }

    div.stButton > button {
        background-color: #0f7f73;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-size: 20px;
        font-weight: 600;
        height: 55px;
    }

    div.stButton > button:hover {
        background-color: #0b665d;
        color: white;
    }

    div[data-testid="stDateInput"] input {
        font-size: 20px;
        height: 52px;
        border-radius: 8px;
    }

    .success-box {
        background: #ecfdf5;
        border: 1px solid #10b981;
        color: #065f46;
        padding: 22px;
        border-radius: 12px;
        text-align: center;
        margin-top: 25px;
        font-size: 22px;
        font-weight: 700;
    }

    .error-box {
        background: #fff1f2;
        border: 1px solid #ff4d4d;
        color: #a00000;
        padding: 18px 22px;
        border-radius: 12px;
        margin-top: 25px;
        font-size: 20px;
    }

    .footer {
        text-align: center;
        color: #5f6f82;
        font-size: 18px;
        margin-top: 35px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["feature_names"], saved["label_encoder"]

model, feature_names, le = load_model()


def get_class_name(class_id):
    return str(le.inverse_transform([int(class_id)])[0])


# -----------------------------
# Weather API with cache
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_weather_features(selected_date):
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

    if response.status_code == 429:
        raise ValueError(
            "Weather API limit reached. Please wait a few minutes and try again."
        )

    response.raise_for_status()
    data = response.json()

    hourly_df = pd.DataFrame(data["hourly"])
    daily_df = pd.DataFrame(data["daily"])

    if hourly_df.empty or daily_df.empty:
        raise ValueError("Weather data is not available for this date.")

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


# -----------------------------
# App UI
# -----------------------------
st.markdown(
    """
    <h1>🌧️ Localized Rainfall Prediction</h1>
    <p class="subtitle">
    Select a date. The system collects weather data automatically and predicts the rainfall class.
    </p>
    """,
    unsafe_allow_html=True
)

today = date.today()
min_date = today
max_date = today + timedelta(days=16)

col1, col2, col3 = st.columns([1.4, 1.1, 1.4])

with col1:
    st.write("")

with col2:
    selected_date = st.date_input(
        "",
        value=today,
        min_value=min_date,
        max_value=max_date
    )

with col3:
    st.write("")

btn_col1, btn_col2, btn_col3 = st.columns([1.45, 1, 1.45])

with btn_col2:
    predict_button = st.button("Get Prediction", use_container_width=True)

st.markdown(
    f"""
    <p class="location-text">
    Location: Dhaka, Bangladesh | Forecast range: {min_date} to {max_date}
    </p>
    """,
    unsafe_allow_html=True
)

if predict_button:
    try:
        with st.spinner("Collecting weather data and predicting rainfall..."):
            weather_values = fetch_weather_features(selected_date)

            input_df = pd.DataFrame([weather_values])
            input_df = input_df[feature_names]

            predicted_id = model.predict(input_df)[0]
            predicted_class = get_class_name(predicted_id)

            confidence = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                confidence = probs.max() * 100

        if confidence is not None:
            st.markdown(
                f"""
                <div class="success-box">
                Predicted Rainfall Class: {predicted_class}<br>
                Confidence: {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="success-box">
                Predicted Rainfall Class: {predicted_class}
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.expander("Weather values used"):
            st.dataframe(
                pd.DataFrame.from_dict(
                    weather_values,
                    orient="index",
                    columns=["Value"]
                ),
                use_container_width=True
            )

    except Exception as error:
        st.markdown(
            f"""
            <div class="error-box">
            <strong>Prediction failed:</strong> {error}
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown(
    """
    <p class="footer">
    Model: Random Forest | Weather API: Open-Meteo | Framework: Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
