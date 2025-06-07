import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "https://weather-api.onrender.com/run"  # Update this after Render deploy

st.set_page_config(page_title="Weather Dashboard", layout="wide")
st.title("🌦️ Real-Time Weather Monitoring Dashboard")

if st.button("Fetch Weather Data"):
    with st.spinner("Getting latest weather info..."):
        res = requests.get(API_URL)
        if res.status_code == 200:
            data = res.json()['results']
            df = pd.DataFrame(data)

            st.subheader("Weather Data")
            st.dataframe(df)

            st.subheader("🌡️ Temperature (°C)")
            fig1 = px.bar(df, x="city", y="temperature_celsius", color="temperature_celsius", text="temperature_celsius")
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("💧 Humidity (%)")
            fig2 = px.line(df, x="city", y="humidity", markers=True)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("Failed to fetch data.")
