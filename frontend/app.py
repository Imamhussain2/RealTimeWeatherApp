import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

API_URL = "https://realtimeweatherapp-8ow1.onrender.com/run"
PREDICT_URL = "https://realtimeweatherapp-8ow1.onrender.com/predict"

st.set_page_config(page_title="🌤️ Real-Time Weather Dashboard", layout="wide")
st.title("🌍 Real-Time Weather Monitoring & ML-based Temperature Prediction")

# ===== Fetch and preprocess weather data =====
def fetch_data():
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            data = response.json()['results']
            df = pd.json_normalize(data)
            df.rename(columns={
                "city": "City",
                "temperature_celsius": "Temperature (°C)",
                "feels_like_celsius": "Feels Like (°C)",
                "temp_min_celsius": "Min Temp (°C)",
                "temp_max_celsius": "Max Temp (°C)",
                "pressure_hpa": "Pressure (hPa)",
                "humidity_percent": "Humidity (%)",
                "wind.speed_m_s": "Wind Speed (m/s)",
                "wind.direction_deg": "Wind Dir (°)",
                "wind.gust_m_s": "Wind Gust (m/s)",
                "clouds_percent": "Clouds (%)",
                "weather.main": "Weather",
                "weather.description": "Weather Description",
                "sunrise_local": "Sunrise",
                "sunset_local": "Sunset",
                "timestamp_local": "Data Time",
                "country": "Country"
            }, inplace=True)
            return df
        else:
            st.error("❌ API Error: Could not retrieve data.")
            return None
    except Exception as e:
        st.error(f"❌ Exception: {e}")
        return None

# Convert Unix to formatted time if needed
def format_time(unixts):
    return datetime.fromtimestamp(unixts).strftime("%H:%M:%S")

# ===== UI Tabs =====
tab1, tab2 = st.tabs(["📊 Weather Analytics", "🤖 ML Prediction"])

# ============ TAB 1: ANALYTICS ============
with tab1:
    st.subheader("📡 Current Weather Stats Across Major Cities")

    if st.button("📥 Load Weather Data"):
        df = fetch_data()
        if df is not None:
            st.success("✅ Data successfully loaded!")
            with st.expander("🔍 Raw Weather Data"):
                st.dataframe(df)

            # Summary Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("🌡️ Avg Temp (°C)", round(df["Temperature (°C)"].mean(), 1))
            col2.metric("💧 Avg Humidity (%)", round(df["Humidity (%)"].mean(), 1))
            col3.metric("🍃 Avg Wind Speed (m/s)", round(df["Wind Speed (m/s)"].mean(), 1))

            # ========== Charts ==========
            st.markdown("### 🔥 Temperature Analysis")
            temp_col1, temp_col2 = st.columns(2)
            with temp_col1:
                fig_temp = px.bar(df, x="City", y="Temperature (°C)", color="Temperature (°C)",
                                  text="Temperature (°C)", color_continuous_scale='thermal')
                st.plotly_chart(fig_temp, use_container_width=True)

            with temp_col2:
                fig_feels = px.bar(df, x="City", y="Feels Like (°C)", color="Feels Like (°C)",
                                   text="Feels Like (°C)", color_continuous_scale='plasma')
                st.plotly_chart(fig_feels, use_container_width=True)

            st.markdown("### 💨 Wind & Clouds Overview")
            wind_col1, wind_col2 = st.columns(2)
            with wind_col1:
                fig_wind = px.bar(df, x="City", y="Wind Speed (m/s)", color="Wind Speed (m/s)",
                                  text="Wind Speed (m/s)", color_continuous_scale='blues')
                st.plotly_chart(fig_wind, use_container_width=True)

            with wind_col2:
                fig_clouds = px.bar(df, x="City", y="Clouds (%)", color="Clouds (%)",
                                    text="Clouds (%)", color_continuous_scale='gray')
                st.plotly_chart(fig_clouds, use_container_width=True)

            st.markdown("### 🌅 Sunrise & Sunset Times")
            df["Sunrise"] = pd.to_datetime(df["Sunrise"])
            df["Sunset"] = pd.to_datetime(df["Sunset"])
            time_df = df[["City", "Sunrise", "Sunset"]]
            st.dataframe(time_df)

            st.markdown("### 🌈 Weather Distribution")
            fig_pie = px.pie(df, names="Weather Description", title="Weather Types",
                             color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("### 🔁 Temperature vs Humidity Scatter")
            fig_scatter = px.scatter(df, x="Temperature (°C)", y="Humidity (%)", color="City",
                                     size="Wind Speed (m/s)", hover_name="City",
                                     color_discrete_sequence=px.colors.qualitative.Set1)
            st.plotly_chart(fig_scatter, use_container_width=True)

# ============ TAB 2: PREDICTION ============
with tab2:
    st.subheader("🔮 Predict Temperature with ML Model")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            city = st.selectbox("Select City", [
                'ahmedabad', 'assam', 'bengaluru', 'chennai', 'delhi',
                'kolkata', 'mumbai', 'panaji', 'pune', 'shimla'
            ])
        with col2:
            humidity = st.slider("Humidity (%)", 0, 100, 50)
        with col3:
            weather = st.selectbox("Weather Condition", [
                "clear sky", "few clouds", "scattered clouds", "broken clouds",
                "shower rain", "rain", "thunderstorm", "snow", "mist"
            ])

        submit = st.form_submit_button("🚀 Predict Temperature")

    if submit:
        payload = {
            "city": city,
            "humidity": humidity,
            "weather": weather
        }

        with st.spinner("Making prediction..."):
            try:
                res = requests.post(PREDICT_URL, json=payload)
                if res.status_code == 200:
                    temp = res.json().get("predicted_temperature")
                    st.success(f"🌡️ Predicted Temperature: {temp:.2f} °C")
                else:
                    st.error(f"Error: {res.status_code}")
            except Exception as e:
                st.error(f"❌ Prediction Failed: {e}")
