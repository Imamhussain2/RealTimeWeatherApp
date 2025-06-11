from fastapi import FastAPI
import requests
import logging
import os
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import datetime

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
CITY_LIST = ['ahmedabad', 'assam', 'bengaluru', 'chennai', 'delhi', 'hyderabad',
             'jaipur', 'kolkata', 'lucknow', 'mumbai', 'panaji', 'pune', 'shimla',
             'srinagar', 'thiruvananthapuram']

logging.basicConfig(level=logging.INFO)

# Load ML model
MODEL_PATH = "model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    logging.info("✅ ML model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load ML model: {e}")
    model = None

class WeatherInput(BaseModel):
    city: str
    humidity: float
    temperature_celsius: float

def unix_to_local_time(unix_time, tz_offset_seconds):
    if unix_time == 0:
        return None
    local_dt = datetime.datetime.utcfromtimestamp(unix_time + tz_offset_seconds)
    return local_dt.strftime('%Y-%m-%d %H:%M:%S')

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
    if not data:
        return None
    main = data.get("main", {})
    wind = data.get("wind", {})
    clouds = data.get("clouds", {})
    sys = data.get("sys", {})
    coord = data.get("coord", {})
    weather = data.get("weather", [{}])[0]
    timezone_offset = data.get("timezone", 0)
    dt = data.get("dt", 0)

    return {
        "city": data.get("name"),
        "coordinates": {"lon": coord.get("lon"), "lat": coord.get("lat")},
        "weather": {"main": weather.get("main"), "description": weather.get("description"), "icon": weather.get("icon")},
        "temperature_celsius": round(main.get("temp", 0) - 273.15, 2),
        "feels_like_celsius": round(main.get("feels_like", 0) - 273.15, 2),
        "temp_min_celsius": round(main.get("temp_min", 0) - 273.15, 2),
        "temp_max_celsius": round(main.get("temp_max", 0) - 273.15, 2),
        "pressure_hpa": main.get("pressure"),
        "humidity_percent": main.get("humidity"),
        "wind": {"speed_m_s": wind.get("speed"), "direction_deg": wind.get("deg"), "gust_m_s": wind.get("gust")},
        "clouds_percent": clouds.get("all"),
        "sunrise_local": unix_to_local_time(sys.get("sunrise", 0), timezone_offset),
        "sunset_local": unix_to_local_time(sys.get("sunset", 0), timezone_offset),
        "timestamp_local": unix_to_local_time(dt, timezone_offset),
        "timezone_offset_seconds": timezone_offset,
        "country": sys.get("country")
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
    return {"message": "Weather pipeline completed successfully.", "results": results}

@app.post("/predict")
def predict_weather_condition(data: WeatherInput):
    if model is None:
        return {"error": "ML model not loaded"}
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)[0]
        return {"predicted_weather_condition": prediction}
    except Exception as e:
        logging.error(f"❌ Prediction error: {e}")
        return {"error": str(e)}
