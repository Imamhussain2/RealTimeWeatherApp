import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

API_URL = "https://realtimeweatherapp-8ow1.onrender.com/run"
PREDICT_URL = "https://realtimeweatherapp-8ow1.onrender.com/predict"

st.set_page_config(page_title="🌤️ Real-Time Weather Dashboard", layout="wide")
st.title("🌍 Real-Time Weather Monitoring & ML-based Weather Classification")

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

# ===== UI Tabs =====
tab1, tab2 = st.tabs(["📊 Weather Analytics", "🤖 ML Weather Classification"])

# ============ TAB 1: ANALYTICS ============
with tab1:
    st.subheader("📡 Current Weather Stats Across Major Cities")

    if st.button("📥 Load Weather Data"):
        df = fetch_data()
        if df is not None:
            st.success("✅ Data successfully loaded!")

            # Raw Data
            with st.expander("🔍 Raw Weather Data"):
                st.dataframe(df)

            # Summary Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("🌡️ Avg Temp (°C)", round(df["Temperature (°C)"].mean(), 1))
            col2.metric("💧 Avg Humidity (%)", round(df["Humidity (%)"].mean(), 1))
            col3.metric("🍃 Avg Wind Speed (m/s)", round(df["Wind Speed (m/s)"].mean(), 1))

            # Convert time column for plots
            df["Data Time"] = pd.to_datetime(df["Data Time"])

            # Temperature Analysis
            st.markdown("### 🔥 Temperature Analysis")
            fig_temp = px.bar(df, x="City", y="Temperature (°C)", color="Temperature (°C)",
                              text="Temperature (°C)", color_continuous_scale='thermal')
            st.plotly_chart(fig_temp, use_container_width=True)

            # Humidity Box Plot
            st.markdown("### 📦 Humidity Distribution by City")
            fig_box = px.box(df, x="City", y="Humidity (%)", color="City")
            st.plotly_chart(fig_box, use_container_width=True)

            # Weather Pie Chart
            st.markdown("### 🌈 Weather Distribution")
            fig_pie = px.pie(df, names="Weather Description", title="Weather Types",
                             color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig_pie, use_container_width=True)

            # Scatter Plot
            st.markdown("### 🔁 Temperature vs Humidity Scatter")
            fig_scatter = px.scatter(df, x="Temperature (°C)", y="Humidity (%)", color="City",
                                     size="Wind Speed (m/s)", hover_name="City",
                                     color_discrete_sequence=px.colors.qualitative.Set1)
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Correlation Heatmap
            st.markdown("### 🧠 Feature Correlation Heatmap")
            numeric_cols = df.select_dtypes(include='number')
            corr = numeric_cols.corr()
            fig_heat = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_heat, use_container_width=True)

            # Wind Rose (Polar plot)
            st.markdown("### 🧭 Wind Direction Intensity")
            df_polar = df[["Wind Speed (m/s)", "Wind Dir (°)", "City"]].dropna()

            # Ensure numeric values and filter out bad data
            df_polar = df_polar[pd.to_numeric(df_polar["Wind Dir (°)"], errors="coerce").notnull()]
            df_polar = df_polar[pd.to_numeric(df_polar["Wind Speed (m/s)"], errors="coerce").notnull()]
            df_polar["Wind Dir (°)"] = df_polar["Wind Dir (°)"].astype(float)
            df_polar["Wind Speed (m/s)"] = df_polar["Wind Speed (m/s)"].astype(float)

            if df_polar.empty:
                st.warning("⚠️ Not enough data to render wind rose chart.")
            else:
                fig_polar = px.bar_polar(df_polar, r="Wind Speed (m/s)", theta="Wind Dir (°)",
                                 color="City", template="plotly_dark")
                st.plotly_chart(fig_polar, use_container_width=True)


            # Country-level weather
            st.markdown("### 🌍 Country-wise Weather Summary")
            fig_country = px.sunburst(df, path=["Country", "City", "Weather"], values="Temperature (°C)",
                                      color="Humidity (%)", color_continuous_scale="Viridis")
            st.plotly_chart(fig_country, use_container_width=True)


# ============ TAB 2: CLASSIFICATION ============
with tab2:
    st.subheader("🔮 Predict Weather Condition using ML")
    with st.form("classification_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            city = st.selectbox("Select City", [
                'ahmedabad', 'assam', 'bengaluru', 'chennai', 'delhi',
                'hyderabad', 'jaipur', 'kolkata', 'lucknow', 'mumbai',
                'panaji', 'pune', 'shimla', 'srinagar', 'thiruvananthapuram'
            ])
        with col2:
            humidity = st.slider("Humidity (%)", 0, 100, 50)
        with col3:
            temperature = st.slider("Temperature (°C)", -10, 50, 25)

        submit = st.form_submit_button("🚀 Classify Weather")

    if submit:
        payload = {
            "city": city,
            "humidity": humidity,
            "temperature_celsius": temperature
        }

        with st.spinner("Classifying weather condition..."):
            try:
                res = requests.post(PREDICT_URL, json=payload)
                if res.status_code == 200:
                    weather = res.json().get("predicted_weather_condition")
                    st.success(f"🌦️ Predicted Weather Condition: {weather}")
                else:
                    st.error(f"Error: {res.status_code}")
            except Exception as e:
                st.error(f"❌ Prediction Failed: {e}")
