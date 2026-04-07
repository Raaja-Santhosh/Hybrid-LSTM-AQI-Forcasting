from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import requests
import joblib
import sys
import json
from datetime import datetime

import database as db

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

OPEN_METEO_AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
SEQUENCE_LENGTH = 30


def fetch_live_aqi(city: str):
    """Fetches LIVE satellite AQI from the Open-Meteo API (free, no key needed)."""
    coords = CITY_COORDS.get(city)
    if not coords:
        return None

    try:
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "current": "pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,us_aqi",
        }
        r = requests.get(OPEN_METEO_AQI_URL, params=params, timeout=5)
        data = r.json().get("current", {})
        return {
            "pm25": data.get("pm2_5"),
            "pm10": data.get("pm10"),
            "no2": data.get("nitrogen_dioxide"),
            "so2": data.get("sulphur_dioxide"),
            "co": data.get("carbon_monoxide"),
            "o3": data.get("ozone"),
            "aqi": data.get("us_aqi"),
        }
    except Exception:
        return None


def fetch_live_weather(city: str):
    """Fetches LIVE weather data (temperature, humidity, wind speed) from Open-Meteo."""
    coords = CITY_COORDS.get(city)
    if not coords:
        return None

    try:
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        }
        r = requests.get(OPEN_METEO_WEATHER_URL, params=params, timeout=5)
        data = r.json().get("current", {})
        return {
            "temperature": data.get("temperature_2m"),
            "humidity": data.get("relative_humidity_2m"),
            "wind_speed": data.get("wind_speed_10m"),
            "weather_code": data.get("weather_code"),
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


def load_model_artifacts():
    model_path = "models/lstm_model.keras"
    scaler_path = "models/scaler.joblib"

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None, None, "Model artifacts not found in models/"

    try:
        from keras import layers
        from keras.models import load_model

        # Compatibility patch for model files containing newer Keras config keys.
        orig_dense_init = layers.Dense.__init__
        orig_input_init = layers.InputLayer.__init__

        def patched_dense_init(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            return orig_dense_init(self, *args, **kwargs)

        def patched_input_init(self, *args, **kwargs):
            kwargs.pop("optional", None)
            if "batch_shape" in kwargs and "batch_input_shape" not in kwargs:
                kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
            return orig_input_init(self, *args, **kwargs)

        layers.Dense.__init__ = patched_dense_init
        layers.InputLayer.__init__ = patched_input_init

        try:
            model = load_model(model_path, compile=False, safe_mode=False)
        finally:
            layers.Dense.__init__ = orig_dense_init
            layers.InputLayer.__init__ = orig_input_init

        # Compatibility alias for artifacts serialized with NumPy internal paths.
        if "numpy._core" not in sys.modules:
            sys.modules["numpy._core"] = np.core
        if "numpy._core.multiarray" not in sys.modules:
            sys.modules["numpy._core.multiarray"] = np.core.multiarray

        scaler = joblib.load(scaler_path)
        return model, scaler, None
    except Exception as exc:
        return None, None, str(exc)


MODEL = None
SCALER = None
MODEL_ERROR = "Not loaded yet"


def load_health_classifier_artifact():
    classifier_path = "models/health_classifier.joblib"
    if not os.path.exists(classifier_path):
        return None, "Health classifier artifact not found in models/"

    try:
        artifact = joblib.load(classifier_path)
        if not isinstance(artifact, dict) or "model" not in artifact or "features" not in artifact:
            return None, "Invalid classifier artifact format"
        return artifact, None
    except Exception as exc:
        return None, str(exc)


HEALTH_CLASSIFIER_ARTIFACT = None
HEALTH_CLASSIFIER_ERROR = "Not loaded yet"


def ensure_lstm_loaded():
    global MODEL, SCALER, MODEL_ERROR
    if MODEL is not None and SCALER is not None:
        return
    MODEL, SCALER, MODEL_ERROR = load_model_artifacts()


def ensure_classifier_loaded():
    global HEALTH_CLASSIFIER_ARTIFACT, HEALTH_CLASSIFIER_ERROR
    if HEALTH_CLASSIFIER_ARTIFACT is not None:
        return
    HEALTH_CLASSIFIER_ARTIFACT, HEALTH_CLASSIFIER_ERROR = load_health_classifier_artifact()


def load_json_file(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def classify_health_ml(features: dict):
    # Runtime classifier inference can be memory-heavy; keep it opt-in.
    if os.getenv("ENABLE_CLASSIFIER_RUNTIME", "0") != "1":
        return None

    ensure_classifier_loaded()
    if HEALTH_CLASSIFIER_ARTIFACT is None:
        return None

    model = HEALTH_CLASSIFIER_ARTIFACT["model"]
    feature_cols = HEALTH_CLASSIFIER_ARTIFACT["features"]

    row = {col: features.get(col, np.nan) for col in feature_cols}
    x_input = pd.DataFrame([row])

    try:
        pred = model.predict(x_input)[0]
        return {
            "level": str(pred),
            "model": str(HEALTH_CLASSIFIER_ARTIFACT.get("model_name", "unknown")),
        }
    except Exception:
        return None


def forecast_multi_horizon(model, scaler, latest_sequence, steps):
    """Iteratively predicts future AQI with mean-reversion to prevent drift."""
    seq = latest_sequence.astype(float).copy().reshape(-1)
    baseline_aqi = float(seq[-1])  # Current AQI as anchor
    predictions = []

    for i in range(steps):
        scaled_seq = scaler.transform(seq.reshape(-1, 1))
        model_input = np.reshape(scaled_seq, (1, len(seq), 1))
        pred_scaled = model.predict(model_input, verbose=0)
        raw_pred = float(scaler.inverse_transform(pred_scaled)[0][0])

        # Mean reversion: blend prediction with baseline (stronger pull for further steps)
        reversion_weight = 0.15 * (i + 1)  # 15%, 30%, 45% pull toward baseline
        pred = raw_pred * (1 - reversion_weight) + baseline_aqi * reversion_weight

        predictions.append(pred)
        seq = np.append(seq[1:], pred)

    return predictions


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
    Now includes weather data and stores readings in MongoDB.
    """
    # Fetch weather in parallel with AQI
    weather = fetch_live_weather(city)
    weather_data = {
        "temperature": weather.get("temperature") if weather else None,
        "humidity": weather.get("humidity") if weather else None,
        "wind_speed": weather.get("wind_speed") if weather else None,
        "weather_code": weather.get("weather_code") if weather else None,
    }

    # Try live AQI data first
    live = fetch_live_aqi(city)

    if live and live.get("aqi") is not None:
        health = classify_aqi(live["aqi"])
        ml_health = classify_health_ml(
            {
                "PM2.5": live.get("pm25"),
                "PM10": live.get("pm10"),
                "NO2": live.get("no2"),
                "SO2": live.get("so2"),
                "City": city,
            }
        )

        # Store reading in MongoDB
        db.store_reading(
            city=city, source="LIVE SATELLITE", aqi=live["aqi"],
            pm25=live.get("pm25"), pm10=live.get("pm10"),
            no2=live.get("no2"), so2=live.get("so2"),
            temperature=weather_data["temperature"],
            humidity=weather_data["humidity"],
            wind_speed=weather_data["wind_speed"],
        )
        if weather:
            db.store_weather(city, weather_data["temperature"],
                            weather_data["humidity"], weather_data["wind_speed"])

        return {
            "source": "LIVE SATELLITE",
            "city": city,
            "aqi": live["aqi"],
            "pm25": live["pm25"],
            "pm10": live["pm10"],
            "no2": live["no2"],
            "so2": live["so2"],
            "co": live.get("co"),
            "o3": live.get("o3"),
            "health_risk": health,
            "health_risk_ml": ml_health,
            "weather": weather_data,
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
        ml_health = classify_health_ml(
            {
                "PM2.5": float(latest.get("PM2.5", np.nan)),
                "PM10": float(latest.get("PM10", np.nan)),
                "NO2": float(latest.get("NO2", np.nan)),
                "SO2": float(latest.get("SO2", np.nan)),
                "CO": float(latest.get("CO", np.nan)) if "CO" in latest else np.nan,
                "O3": float(latest.get("O3", np.nan)) if "O3" in latest else np.nan,
                "City": city,
            }
        )
        return {
            "source": "CSV FALLBACK",
            "city": city,
            "aqi": float(latest['AQI']),
            "pm25": float(latest.get('PM2.5', 0)),
            "pm10": float(latest.get('PM10', 0)),
            "no2": float(latest.get('NO2', 0)),
            "health_risk": health,
            "health_risk_ml": ml_health,
            "weather": weather_data,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/predict")
def run_lstm_prediction(city: str = "Delhi"):
    """
    Hybrid Prediction: 29 days from CSV + 1 LIVE day injected as Day 30.
    Returns 24h/48h/72h forecasts.
    """
    try:
        ensure_lstm_loaded()
        if MODEL is None or SCALER is None:
            return {
                "status": "MODEL_UNAVAILABLE",
                "error": MODEL_ERROR or "Model artifacts not loaded.",
            }

        df = pd.read_csv("data/master_dataset.csv", low_memory=False)
        city_data = df[df['City'] == city].dropna(subset=['AQI'])

        # Grab 29 historical days from CSV
        last_29_days = city_data.tail(SEQUENCE_LENGTH - 1)[['AQI']].values.tolist()

        if len(last_29_days) < SEQUENCE_LENGTH - 1:
            return {"error": f"Not enough data for {city}."}

        # Inject LIVE Day 30 from satellite
        live = fetch_live_aqi(city)
        if live and live.get("aqi") is not None:
            last_29_days.append([float(live["aqi"])])
            data_source = "HYBRID (29 CSV + 1 LIVE)"
        else:
            # If satellite is down, use the 30th CSV row instead
            fallback = city_data.tail(SEQUENCE_LENGTH)[['AQI']].values.tolist()
            if len(fallback) < SEQUENCE_LENGTH:
                return {"error": f"Not enough fallback data for {city}."}
            last_29_days = fallback
            data_source = "CSV ONLY (satellite unreachable)"

        raw_aqi = np.array(last_29_days[-SEQUENCE_LENGTH:], dtype=float)

        # Predict 3 daily steps (not 72 hourly — the model was trained on daily data)
        future_preds = forecast_multi_horizon(MODEL, SCALER, raw_aqi, steps=3)

        pred_24h = float(future_preds[0])  # Day +1
        pred_48h = float(future_preds[1])  # Day +2
        pred_72h = float(future_preds[2])  # Day +3
        health_24h = classify_aqi(pred_24h)
        health_48h = classify_aqi(pred_48h)
        health_72h = classify_aqi(pred_72h)

        # Store prediction in MongoDB
        db.store_prediction(
            city=city, data_source=data_source,
            pred_24h=round(pred_24h, 2),
            pred_48h=round(pred_48h, 2),
            pred_72h=round(pred_72h, 2),
        )

        return {
            "status": "LIVE",
            "data_source": data_source,
            # Backward compatibility with older clients.
            "predicted_aqi_tomorrow": round(pred_24h, 2),
            "health_risk": health_24h,
            "predictions": {
                "24h": {
                    "aqi": round(pred_24h, 2),
                    "health_risk": health_24h,
                },
                "48h": {
                    "aqi": round(pred_48h, 2),
                    "health_risk": health_48h,
                },
                "72h": {
                    "aqi": round(pred_72h, 2),
                    "health_risk": health_72h,
                },
            },
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/api/classifier-status")
def classifier_status():
    classifier_file_exists = os.path.exists("models/health_classifier.joblib")
    return {
        "loaded": HEALTH_CLASSIFIER_ARTIFACT is not None,
        "artifact_exists": classifier_file_exists,
        "runtime_enabled": os.getenv("ENABLE_CLASSIFIER_RUNTIME", "0") == "1",
        "error": HEALTH_CLASSIFIER_ERROR if HEALTH_CLASSIFIER_ARTIFACT is None else None,
    }


@app.get("/api/model-metrics")
def model_metrics():
    classifier_metrics = load_json_file("models/health_classifier_metrics.json")
    lstm_metrics = load_json_file("models/lstm_training_metrics.json")
    arima_metrics = load_json_file("models/arima_baseline_metrics.json")
    classifier_file_exists = os.path.exists("models/health_classifier.joblib")
    lstm_model_exists = os.path.exists("models/lstm_model.keras") and os.path.exists("models/scaler.joblib")

    return {
        "lstm": {
            "artifact_loaded": MODEL is not None,
            "artifact_exists": lstm_model_exists,
            "load_error": MODEL_ERROR if MODEL is None else None,
            "metrics": lstm_metrics,
        },
        "classifier": {
            "artifact_loaded": HEALTH_CLASSIFIER_ARTIFACT is not None,
            "artifact_exists": classifier_file_exists,
            "runtime_enabled": os.getenv("ENABLE_CLASSIFIER_RUNTIME", "0") == "1",
            "load_error": HEALTH_CLASSIFIER_ERROR if HEALTH_CLASSIFIER_ARTIFACT is None else None,
            "selected_model": HEALTH_CLASSIFIER_ARTIFACT.get("model_name") if HEALTH_CLASSIFIER_ARTIFACT else None,
            "metrics": classifier_metrics,
        },
        "arima_baseline": {
            "metrics": arima_metrics,
            "available": arima_metrics is not None,
        },
    }


@app.get("/api/report-summary")
def report_summary(city: str = "Delhi"):
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "city": city,
        "current_status": get_current_status(city),
        "forecast": run_lstm_prediction(city),
        "model_metrics": model_metrics(),
    }


@app.post("/api/report-summary/export")
def export_report_summary(city: str = "Delhi"):
    report = report_summary(city)
    os.makedirs("reports", exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_name = f"report_summary_{city}_{stamp}.json".replace(" ", "_")
    file_path = os.path.join("reports", file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return {
        "status": "saved",
        "file": file_path,
        "report": report,
    }


@app.get("/api/historical")
def get_historical(city: str = "Delhi", days: int = 90):
    """Returns historical AQI data for trend charts."""
    if not os.path.exists("data/master_dataset.csv"):
        return {"error": "Dataset not found"}

    try:
        df = pd.read_csv("data/master_dataset.csv", low_memory=False)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        city_data = df[df['City'] == city].dropna(subset=['AQI', 'Date'])
        city_data = city_data.sort_values('Date').tail(days)

        records = []
        for _, row in city_data.iterrows():
            records.append({
                "date": row['Date'].strftime('%Y-%m-%d'),
                "aqi": float(row['AQI']),
                "pm25": float(row.get('PM2.5', 0)) if pd.notna(row.get('PM2.5')) else None,
                "pm10": float(row.get('PM10', 0)) if pd.notna(row.get('PM10')) else None,
                "no2": float(row.get('NO2', 0)) if pd.notna(row.get('NO2')) else None,
            })

        return {
            "city": city,
            "days_requested": days,
            "records_returned": len(records),
            "data": records,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/compare")
def compare_cities(cities: str = "Delhi,Mumbai,Bangalore"):
    """Fetches current AQI for multiple cities for side-by-side comparison."""
    city_list = [c.strip() for c in cities.split(",") if c.strip()]
    results = []

    for city in city_list[:5]:  # Max 5 cities
        live = fetch_live_aqi(city)
        if live and live.get("aqi") is not None:
            health = classify_aqi(live["aqi"])
            results.append({
                "city": city,
                "source": "LIVE SATELLITE",
                "aqi": live["aqi"],
                "pm25": live["pm25"],
                "pm10": live["pm10"],
                "no2": live["no2"],
                "so2": live["so2"],
                "health_risk": health,
            })
        else:
            # Fallback to CSV
            try:
                df = pd.read_csv("data/master_dataset.csv", low_memory=False)
                city_data = df[df['City'] == city].dropna(subset=['AQI'])
                if not city_data.empty:
                    latest = city_data.iloc[-1]
                    health = classify_aqi(float(latest['AQI']))
                    results.append({
                        "city": city,
                        "source": "CSV FALLBACK",
                        "aqi": float(latest['AQI']),
                        "pm25": float(latest.get('PM2.5', 0)),
                        "pm10": float(latest.get('PM10', 0)),
                        "no2": float(latest.get('NO2', 0)),
                        "so2": float(latest.get('SO2', 0)) if 'SO2' in latest else 0,
                        "health_risk": health,
                    })
            except Exception:
                results.append({"city": city, "error": "Data unavailable"})

    return {"cities": results}


from fastapi.responses import Response as RawResponse
import io

@app.get("/api/export-csv")
def export_csv(city: str = "Delhi", days: int = 90):
    """Downloads a CSV file of historical AQI data for the selected city."""
    if not os.path.exists("data/master_dataset.csv"):
        return {"error": "Dataset not found"}

    try:
        df = pd.read_csv("data/master_dataset.csv", low_memory=False)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        city_data = df[df['City'] == city].dropna(subset=['AQI', 'Date'])
        city_data = city_data.sort_values('Date').tail(days)

        cols = ['Date', 'City', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
        available_cols = [c for c in cols if c in city_data.columns]
        export_df = city_data[available_cols].copy()
        export_df['Date'] = export_df['Date'].dt.strftime('%Y-%m-%d')

        csv_content = export_df.to_csv(index=False)
        filename = f"aqi_report_{city}_{days}days.csv"

        return RawResponse(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": "text/csv; charset=utf-8",
            },
        )
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────
# Weather & Database Endpoints
# ──────────────────────────────────────────────
@app.get("/api/weather")
def get_weather(city: str = "Delhi"):
    """Returns current weather conditions for the selected city."""
    weather = fetch_live_weather(city)
    if weather is None:
        return {"error": f"Weather data unavailable for {city}"}

    db.store_weather(city, weather.get("temperature"),
                     weather.get("humidity"), weather.get("wind_speed"))

    return {
        "city": city,
        "temperature": weather.get("temperature"),
        "humidity": weather.get("humidity"),
        "wind_speed": weather.get("wind_speed"),
        "weather_code": weather.get("weather_code"),
    }


@app.get("/api/db-status")
def db_status():
    """Returns MongoDB connection status and collection counts."""
    return db.get_db_stats()


@app.get("/api/readings")
def get_stored_readings(city: str = None, limit: int = 50):
    """Returns stored AQI readings from MongoDB."""
    readings = db.get_readings(city=city, limit=limit)
    # Convert datetime objects for JSON serialization
    for r in readings:
        if "timestamp" in r and hasattr(r["timestamp"], "isoformat"):
            r["timestamp"] = r["timestamp"].isoformat()
    return {"readings": readings, "count": len(readings)}


@app.get("/api/stored-predictions")
def get_stored_predictions(city: str = None, limit: int = 20):
    """Returns stored LSTM predictions from MongoDB."""
    preds = db.get_predictions(city=city, limit=limit)
    for p in preds:
        if "timestamp" in p and hasattr(p["timestamp"], "isoformat"):
            p["timestamp"] = p["timestamp"].isoformat()
    return {"predictions": preds, "count": len(preds)}


# ──────────────────────────────────────────────
# Serve Frontend (must be LAST)
# ──────────────────────────────────────────────
@app.get("/app")
@app.get("/app/{full_path:path}")
def serve_frontend(full_path: str = ""):
    """Serves the Sentinel AQI dashboard at /app."""
    return FileResponse("frontend/index.html")

if os.path.isdir("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

