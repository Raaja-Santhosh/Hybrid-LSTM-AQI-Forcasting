"""
MongoDB Database Layer for AtmosIQ
------------------------------------
Stores AQI readings and predictions in a local MongoDB instance.
Falls back gracefully if MongoDB is unreachable (system continues to work with CSV).

Collections:
  - aqi_readings   : real-time and historical AQI observations
  - predictions     : LSTM forecast results
  - weather_logs    : weather telemetry snapshots
"""

import os
from datetime import datetime, timezone

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB", "atmosiq")

_client = None
_db = None
_connected = False


def get_db():
    """Returns the MongoDB database handle, or None if unavailable."""
    global _client, _db, _connected

    if not PYMONGO_AVAILABLE:
        return None

    if _db is not None and _connected:
        return _db

    try:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        # Force a connection check
        _client.admin.command("ping")
        _db = _client[DB_NAME]
        _connected = True
        print(f"✅ MongoDB connected: {MONGO_URI} / {DB_NAME}")
        return _db
    except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as exc:
        print(f"⚠️  MongoDB unavailable ({exc}). Running in CSV-only mode.")
        _connected = False
        return None


def is_connected() -> bool:
    """Check if MongoDB is currently connected."""
    return _connected and _db is not None


# ──────────────────────────────────────────────
# AQI Readings
# ──────────────────────────────────────────────
def store_reading(city: str, source: str, aqi: float, pm25: float = None,
                  pm10: float = None, no2: float = None, so2: float = None,
                  temperature: float = None, humidity: float = None,
                  wind_speed: float = None):
    """Stores an AQI reading snapshot in MongoDB."""
    db = get_db()
    if db is None:
        return None

    doc = {
        "city": city,
        "source": source,
        "aqi": aqi,
        "pm25": pm25,
        "pm10": pm10,
        "no2": no2,
        "so2": so2,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "timestamp": datetime.now(timezone.utc),
    }

    try:
        result = db.aqi_readings.insert_one(doc)
        return str(result.inserted_id)
    except Exception:
        return None


def get_readings(city: str = None, limit: int = 50):
    """Retrieves recent AQI readings from MongoDB."""
    db = get_db()
    if db is None:
        return []

    try:
        query = {"city": city} if city else {}
        cursor = db.aqi_readings.find(
            query,
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit)
        return list(cursor)
    except Exception:
        return []


# ──────────────────────────────────────────────
# Predictions
# ──────────────────────────────────────────────
def store_prediction(city: str, data_source: str, pred_24h: float,
                     pred_48h: float, pred_72h: float):
    """Stores an LSTM prediction result in MongoDB."""
    db = get_db()
    if db is None:
        return None

    doc = {
        "city": city,
        "data_source": data_source,
        "pred_24h": pred_24h,
        "pred_48h": pred_48h,
        "pred_72h": pred_72h,
        "timestamp": datetime.now(timezone.utc),
    }

    try:
        result = db.predictions.insert_one(doc)
        return str(result.inserted_id)
    except Exception:
        return None


def get_predictions(city: str = None, limit: int = 20):
    """Retrieves recent predictions from MongoDB."""
    db = get_db()
    if db is None:
        return []

    try:
        query = {"city": city} if city else {}
        cursor = db.predictions.find(
            query,
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit)
        return list(cursor)
    except Exception:
        return []


# ──────────────────────────────────────────────
# Weather Logs
# ──────────────────────────────────────────────
def store_weather(city: str, temperature: float, humidity: float,
                  wind_speed: float):
    """Stores a weather telemetry snapshot in MongoDB."""
    db = get_db()
    if db is None:
        return None

    doc = {
        "city": city,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "timestamp": datetime.now(timezone.utc),
    }

    try:
        result = db.weather_logs.insert_one(doc)
        return str(result.inserted_id)
    except Exception:
        return None


# ──────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────
def get_db_stats():
    """Returns database statistics for the status endpoint."""
    db = get_db()
    if db is None:
        return {
            "connected": False,
            "message": "MongoDB unavailable — running in CSV-only mode",
        }

    try:
        return {
            "connected": True,
            "database": DB_NAME,
            "uri": MONGO_URI,
            "collections": {
                "aqi_readings": db.aqi_readings.count_documents({}),
                "predictions": db.predictions.count_documents({}),
                "weather_logs": db.weather_logs.count_documents({}),
            },
        }
    except Exception as exc:
        return {
            "connected": False,
            "message": str(exc),
        }
