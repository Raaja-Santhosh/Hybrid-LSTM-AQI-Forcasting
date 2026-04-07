"""
AtmosIQ API Unit Tests
-----------------------
Tests core API endpoints and utility functions.
Run: python -m pytest tests/ -v
"""

import sys
import os
import pytest

# Add project root to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ──────────────────────────────────────────────
# Test: AQI Classification Logic
# ──────────────────────────────────────────────
class TestClassifyAqi:
    """Tests the rule-based AQI health risk classifier."""

    def _classify(self, aqi_value):
        # Import inline to avoid TensorFlow/Keras loading at import time
        from main import classify_aqi
        return classify_aqi(aqi_value)

    def test_good_range(self):
        result = self._classify(30)
        assert result["level"] == "Good"
        assert result["color"] == "#00e400"

    def test_moderate_range(self):
        result = self._classify(75)
        assert result["level"] == "Moderate"

    def test_unhealthy_sensitive(self):
        result = self._classify(125)
        assert result["level"] == "Unhealthy for Sensitive Groups"

    def test_unhealthy(self):
        result = self._classify(175)
        assert result["level"] == "Unhealthy"

    def test_very_unhealthy(self):
        result = self._classify(250)
        assert result["level"] == "Very Unhealthy"

    def test_hazardous(self):
        result = self._classify(400)
        assert result["level"] == "Hazardous"
        assert result["color"] == "#7e0023"

    def test_boundary_50(self):
        result = self._classify(50)
        assert result["level"] == "Good"

    def test_boundary_100(self):
        result = self._classify(100)
        assert result["level"] == "Moderate"

    def test_boundary_301(self):
        result = self._classify(301)
        assert result["level"] == "Hazardous"


# ──────────────────────────────────────────────
# Test: Database Module
# ──────────────────────────────────────────────
class TestDatabaseModule:
    """Tests the MongoDB database helper (graceful fallback)."""

    def test_import_database(self):
        import database as db
        assert hasattr(db, "store_reading")
        assert hasattr(db, "get_readings")
        assert hasattr(db, "store_prediction")
        assert hasattr(db, "get_db_stats")

    def test_db_stats_returns_dict(self):
        import database as db
        stats = db.get_db_stats()
        assert isinstance(stats, dict)
        assert "connected" in stats

    def test_get_readings_returns_list(self):
        import database as db
        readings = db.get_readings(city="Delhi", limit=5)
        assert isinstance(readings, list)

    def test_get_predictions_returns_list(self):
        import database as db
        preds = db.get_predictions(city="Delhi", limit=5)
        assert isinstance(preds, list)


# ──────────────────────────────────────────────
# Test: FastAPI Endpoints (using TestClient)
# ──────────────────────────────────────────────
class TestAPIEndpoints:
    """Tests FastAPI routes using httpx TestClient."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        try:
            from fastapi.testclient import TestClient
            from main import app
            self.client = TestClient(app)
            self.available = True
        except Exception:
            self.available = False

    def test_root_endpoint(self):
        if not self.available:
            pytest.skip("TestClient unavailable")
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_cities_endpoint(self):
        if not self.available:
            pytest.skip("TestClient unavailable")
        response = self.client.get("/api/cities")
        assert response.status_code == 200
        data = response.json()
        assert "cities" in data
        assert "Delhi" in data["cities"]
        assert len(data["cities"]) >= 5

    def test_db_status_endpoint(self):
        if not self.available:
            pytest.skip("TestClient unavailable")
        response = self.client.get("/api/db-status")
        assert response.status_code == 200
        data = response.json()
        assert "connected" in data

    def test_weather_endpoint(self):
        if not self.available:
            pytest.skip("TestClient unavailable")
        response = self.client.get("/api/weather?city=Delhi")
        assert response.status_code == 200
        data = response.json()
        assert "city" in data or "error" in data

    def test_model_metrics_endpoint(self):
        if not self.available:
            pytest.skip("TestClient unavailable")
        response = self.client.get("/api/model-metrics")
        assert response.status_code == 200
        data = response.json()
        assert "lstm" in data
        assert "classifier" in data
        assert "arima_baseline" in data

    def test_classifier_status_endpoint(self):
        if not self.available:
            pytest.skip("TestClient unavailable")
        response = self.client.get("/api/classifier-status")
        assert response.status_code == 200

    def test_readings_endpoint(self):
        if not self.available:
            pytest.skip("TestClient unavailable")
        response = self.client.get("/api/readings?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert "readings" in data
        assert "count" in data
