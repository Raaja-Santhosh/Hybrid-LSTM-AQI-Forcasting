from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import requests

app = FastAPI(title="AtmosIQ API", description="Hybrid LSTM-Based AQI Forecasting & Health Risk System")

# CORS - allows the React frontend to call this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# City coordinates for the Live API
# ──────────────────────────────────────────────
CITY_COORDS = {
    "Delhi":     {"lat": 28.6139, "lon": 77.2090},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Chennai":   {"lat": 13.0827, "lon": 80.2707},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Pune":      {"lat": 18.5204, "lon": 73.8567},
    "Lucknow":   {"lat": 26.8467, "lon": 80.9462},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "Jaipur":    {"lat": 26.9124, "lon": 75.7873},
}

OPEN_METEO_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def fetch_live_aqi(city: str):
    """Fetches LIVE satellite AQI from the Open-Meteo API (free, no key needed)."""
    coords = CITY_COORDS.get(city)
    if not coords:
        return None

    try:
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "current": "pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,us_aqi",
        }
        r = requests.get(OPEN_METEO_URL, params=params, timeout=5)
        data = r.json().get("current", {})
        return {
            "pm25": data.get("pm2_5"),
            "pm10": data.get("pm10"),
            "no2": data.get("nitrogen_dioxide"),
            "so2": data.get("sulphur_dioxide"),
            "aqi": data.get("us_aqi"),
        }
    except Exception:
        return None


# ──────────────────────────────────────────────
# Health classification helper
# ──────────────────────────────────────────────
def classify_aqi(aqi_value):
    """Converts a raw AQI number into a human-readable health risk level."""
    if aqi_value <= 50:
        return {"level": "Good", "color": "#00e400", "advice": "Air quality is satisfactory. Enjoy outdoor activities."}
    elif aqi_value <= 100:
        return {"level": "Moderate", "color": "#ffff00", "advice": "Acceptable. Sensitive individuals should limit prolonged outdoor exertion."}
    elif aqi_value <= 150:
        return {"level": "Unhealthy for Sensitive Groups", "color": "#ff7e00", "advice": "People with asthma or heart conditions should reduce outdoor activity."}
    elif aqi_value <= 200:
        return {"level": "Unhealthy", "color": "#ff0000", "advice": "Everyone may experience health effects. Wear N95 mask outdoors."}
    elif aqi_value <= 300:
        return {"level": "Very Unhealthy", "color": "#8f3f97", "advice": "Health alert: significant risk. Avoid all outdoor exertion."}
    else:
        return {"level": "Hazardous", "color": "#7e0023", "advice": "Emergency conditions. Stay indoors. Keep windows sealed."}


# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "AtmosIQ Backend API is officially online!", "version": "2.0"}


@app.get("/api/cities")
def get_cities():
    """Returns list of all supported cities for the frontend dropdown."""
    return {"cities": list(CITY_COORDS.keys())}


@app.get("/api/current-status")
def get_current_status(city: str = "Delhi"):
    """
    Called by the Dashboard screen.
    PRIORITY: Live satellite data first. Falls back to CSV if satellite is down.
    """
    # Try live data first
    live = fetch_live_aqi(city)

    if live and live.get("aqi") is not None:
        health = classify_aqi(live["aqi"])
        return {
            "source": "LIVE SATELLITE",
            "city": city,
            "aqi": live["aqi"],
            "pm25": live["pm25"],
            "pm10": live["pm10"],
            "no2": live["no2"],
            "so2": live["so2"],
            "health_risk": health,
        }

    # Fallback to CSV if satellite is unreachable
    if not os.path.exists("data/master_dataset.csv"):
        return {"error": "No live data and no CSV dataset found."}

    try:
        df = pd.read_csv("data/master_dataset.csv", low_memory=False)
        city_data = df[df['City'] == city].dropna(subset=['AQI'])
        if city_data.empty:
            return {"error": f"No data for {city}"}
        latest = city_data.iloc[-1]
        health = classify_aqi(float(latest['AQI']))
        return {
            "source": "CSV FALLBACK",
            "city": city,
            "aqi": float(latest['AQI']),
            "pm25": float(latest.get('PM2.5', 0)),
            "pm10": float(latest.get('PM10', 0)),
            "no2": float(latest.get('NO2', 0)),
            "health_risk": health,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/predict")
def run_lstm_prediction(city: str = "Delhi"):
    """
    Hybrid Prediction: 29 days from CSV + 1 LIVE day injected as Day 30.
    """
    try:
        df = pd.read_csv("data/master_dataset.csv", low_memory=False)
        city_data = df[df['City'] == city].dropna(subset=['AQI'])

        # Grab 29 historical days from CSV
        last_29_days = city_data.tail(29)[['AQI']].values.tolist()

        if len(last_29_days) < 29:
            return {"error": f"Not enough data for {city}."}

        # Inject LIVE Day 30 from satellite
        live = fetch_live_aqi(city)
        if live and live.get("aqi") is not None:
            last_29_days.append([float(live["aqi"])])
            data_source = "HYBRID (29 CSV + 1 LIVE)"
        else:
            # If satellite is down, use the 30th CSV row instead
            fallback = city_data.tail(30)[['AQI']].values.tolist()
            last_29_days = fallback
            data_source = "CSV ONLY (satellite unreachable)"

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        raw_aqi = np.array(last_29_days[-30:])  # ensure exactly 30
        scaled_aqi = scaler.fit_transform(raw_aqi)
        ai_input = np.reshape(scaled_aqi, (1, 30, 1))

        model_path = 'models/lstm_model.keras'

        if os.path.exists(model_path):
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            prediction_scaled = model.predict(ai_input)
            predicted_aqi = scaler.inverse_transform(prediction_scaled)[0][0]
            health = classify_aqi(float(predicted_aqi))

            return {
                "status": "LIVE",
                "data_source": data_source,
                "predicted_aqi_tomorrow": round(float(predicted_aqi), 2),
                "health_risk": health,
            }
        else:
            return {
                "status": "MOCK",
                "data_source": data_source,
                "predicted_aqi_tomorrow": 182.4,
                "health_risk": classify_aqi(182.4),
                "note": "Waiting for lstm_model.keras in models/ folder.",
            }

    except Exception as e:
        return {"error": str(e)}
