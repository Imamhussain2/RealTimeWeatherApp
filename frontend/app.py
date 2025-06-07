import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "https://realtimeweatherapp-8ow1.onrender.com/run"  # Your backend endpoint

st.set_page_config(page_title="Weather Dashboard", layout="wide")
st.title("🌦️ Real-Time Weather Monitoring Dashboard")

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
            fig1 = px.bar(df, x="city", y="temperature_celsius", color="temperature_celsius", text="temperature_celsius")
            st.plotly_chart(fig1, use_container_width=True)

            # Humidity Line Chart
            st.subheader("💧 Humidity by City (%)")
            fig2 = px.line(df, x="city", y="humidity", markers=True)
            st.plotly_chart(fig2, use_container_width=True)

            # Humidity vs Temperature Scatter
            st.subheader("🔁 Temperature vs Humidity")
            fig3 = px.scatter(df, x="temperature_celsius", y="humidity", color="city", size="humidity", hover_name="city")
            st.plotly_chart(fig3, use_container_width=True)

            # Weather type distribution
            st.subheader("🌥️ Weather Type Distribution")
            fig4 = px.pie(df, names="weather", title="Current Weather Conditions")
            st.plotly_chart(fig4, use_container_width=True)

            # Interactive Temperature Range Filter
            st.subheader("🎛️ Filter by Temperature Range")
            min_temp, max_temp = float(df['temperature_celsius'].min()), float(df['temperature_celsius'].max())
            temp_range = st.slider("Select Temperature Range (°C)", min_value=min_temp, max_value=max_temp, value=(min_temp, max_temp))
            filtered_df = df[(df['temperature_celsius'] >= temp_range[0]) & (df['temperature_celsius'] <= temp_range[1])]
            st.dataframe(filtered_df)

        else:
            st.error("❌ Failed to fetch data from the backend.")
