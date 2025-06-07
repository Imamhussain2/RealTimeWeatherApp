import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "https://realtimeweatherapp-8ow1.onrender.com/run"         # Your backend endpoint for weather data
PREDICT_URL = "https://realtimeweatherapp-8ow1.onrender.com/predict" # Your backend endpoint for prediction

st.set_page_config(page_title="Weather Dashboard with ML Predictions", layout="wide")
st.title("🌦️ Real-Time Weather Monitoring & Temperature Prediction Dashboard")

# ======== Fetch and display real-time weather data ========
if st.button("Fetch Weather Data"):
    with st.spinner("Getting latest weather info..."):
        res = requests.get(API_URL)
        if res.status_code == 200:
            data = res.json()['results']
            df = pd.DataFrame(data)

            st.subheader("📋 Raw Weather Data")
            st.dataframe(df)

            # Summary statistics
            st.subheader("📊 Summary Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Temperature (°C)", round(df['temperature_celsius'].mean(), 2))
                st.metric("Max Temperature (°C)", df['temperature_celsius'].max())
                st.metric("Min Temperature (°C)", df['temperature_celsius'].min())
            with col2:
                st.metric("Avg Humidity (%)", round(df['humidity'].mean(), 2))
                st.metric("Max Humidity (%)", df['humidity'].max())
                st.metric("Min Humidity (%)", df['humidity'].min())

            # Hottest and coldest cities
            hottest = df.loc[df['temperature_celsius'].idxmax()]
            coldest = df.loc[df['temperature_celsius'].idxmin()]
            st.info(f"🔥 Hottest City: {hottest['city']} ({hottest['temperature_celsius']}°C)")
            st.info(f"❄️ Coldest City: {coldest['city']} ({coldest['temperature_celsius']}°C)")

            # Temperature Bar Chart
            st.subheader("🌡️ Temperature by City (°C)")
            fig1 = px.bar(df, x="city", y="temperature_celsius", color="temperature_celsius", text="temperature_celsius",
                          color_continuous_scale='thermal')
            st.plotly_chart(fig1, use_container_width=True)

            # Humidity Line Chart
            st.subheader("💧 Humidity by City (%)")
            fig2 = px.line(df, x="city", y="humidity", markers=True, color_discrete_sequence=['blue'])
            st.plotly_chart(fig2, use_container_width=True)

            # Humidity vs Temperature Scatter
            st.subheader("🔁 Temperature vs Humidity")
            fig3 = px.scatter(df, x="temperature_celsius", y="humidity", color="city", size="humidity", hover_name="city",
                              color_continuous_scale=px.colors.sequential.Viridis)
            st.plotly_chart(fig3, use_container_width=True)

            # Weather type distribution
            st.subheader("🌥️ Weather Type Distribution")
            fig4 = px.pie(df, names="weather", title="Current Weather Conditions", color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig4, use_container_width=True)

            # Interactive Temperature Range Filter
            st.subheader("🎛️ Filter by Temperature Range")
            min_temp, max_temp = float(df['temperature_celsius'].min()), float(df['temperature_celsius'].max())
            temp_range = st.slider("Select Temperature Range (°C)", min_value=min_temp, max_value=max_temp, value=(min_temp, max_temp))
            filtered_df = df[(df['temperature_celsius'] >= temp_range[0]) & (df['temperature_celsius'] <= temp_range[1])]
            st.dataframe(filtered_df)

        else:
            st.error("❌ Failed to fetch data from the backend.")

st.markdown("---")

# ======== ML Prediction Section ========
st.subheader("🤖 Predict Temperature Based on City, Humidity & Weather")

# Possible weather conditions from your dataset (you can adjust this list if needed)
weather_options = [
    "clear sky", "few clouds", "scattered clouds", "broken clouds",
    "shower rain", "rain", "thunderstorm", "snow", "mist"
]

with st.form(key='prediction_form'):
    col1, col2, col3 = st.columns(3)

    with col1:
        city_input = st.selectbox("Select City", options=[
            'ahmedabad', 'assam', 'bengaluru', 'chennai', 'delhi',
            'kolkata', 'mumbai', 'panaji', 'pune', 'shimla'
        ])

    with col2:
        humidity_input = st.slider("Humidity (%)", min_value=0, max_value=100, value=50)

    with col3:
        weather_input = st.selectbox("Weather Condition", options=weather_options)

    submit_btn = st.form_submit_button(label='Predict Temperature')

if submit_btn:
    payload = {
        "city": city_input,
        "humidity": humidity_input,
        "weather": weather_input
    }

    with st.spinner("Predicting temperature..."):
        try:
            response = requests.post(PREDICT_URL, json=payload)
            if response.status_code == 200:
                predicted_temp = response.json().get("predicted_temperature_celsius")
                st.success(f"🌡️ Predicted Temperature: {predicted_temp:.2f} °C")
            else:
                st.error(f"Prediction failed with status code {response.status_code}. Please try again.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
