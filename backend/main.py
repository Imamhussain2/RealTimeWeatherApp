from fastapi import FastAPI
import requests
import logging
import os
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY", "e14afbb524e1dd6656dc7ac8eb3b09df")
CITY_LIST = ['ahmedabad', 'assam', 'bengaluru', 'chennai', 'delhi', 'kolkata', 'mumbai', 'panaji', 'pune', 'shimla']

logging.basicConfig(level=logging.INFO)

# Load ML model
MODEL_PATH = "model.pkl"  # Ensure this model exists in the same directory
try:
    model = joblib.load(MODEL_PATH)
    logging.info("✅ ML model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load ML model: {e}")
    model = None

class WeatherInput(BaseModel):
    city: str
    humidity: float

def get_weather_data(city: str):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            logging.info(f"✅ Success: Weather data fetched for {city}")
            return response.json()
        else:
            logging.warning(f"❌ Failed: Status code {response.status_code} for {city}")
            return None
    except Exception as e:
        logging.error(f"❌ Error fetching data for {city}: {e}")
        return None

def process_weather_data(data):
    if not data: return None
    return {
        "city": data.get("name"),
        "temperature_celsius": round(data["main"]["temp"] - 273.15, 2),
        "humidity": data["main"]["humidity"],
        "weather": data["weather"][0]["description"]
    }

@app.get("/")
def read_root():
    return {"message": "✅ FastAPI backend is live. Use /run to fetch weather data."}

@app.get("/run")
def run_pipeline():
    results: List[dict] = []
    for city in CITY_LIST:
        raw = get_weather_data(city)
        processed = process_weather_data(raw)
        if processed:
            results.append(processed)

    logging.info("✅ Pipeline completed")
    return {
        "message": "Weather pipeline completed successfully.",
        "results": results
    }

@app.post("/predict")
def predict_temperature(data: WeatherInput):
    if model is None:
        return {"error": "ML model not loaded"}
    try:
        input_df = pd.DataFrame([data.dict()])
        input_df = pd.get_dummies(input_df).reindex(columns=model.feature_names_in_, fill_value=0)
        prediction = model.predict(input_df)[0]
        return {"predicted_temperature": round(prediction, 2)}
    except Exception as e:
        logging.error(f"❌ Prediction error: {e}")
        return {"error": str(e)}
