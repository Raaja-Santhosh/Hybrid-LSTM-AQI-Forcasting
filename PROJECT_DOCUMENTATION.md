# Hybrid LSTM-Based AQI Forecasting and Health Risk Classification

## Project Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Project Structure](#4-project-structure)
5. [Data Pipeline](#5-data-pipeline)
6. [Machine Learning Models](#6-machine-learning-models)
7. [Backend API (FastAPI)](#7-backend-api-fastapi)
8. [Frontend Dashboard](#8-frontend-dashboard)
9. [Database Layer (MongoDB)](#9-database-layer-mongodb)
10. [IoT Simulation](#10-iot-simulation)
11. [Testing](#11-testing)
12. [Results and Evaluation](#12-results-and-evaluation)
13. [How to Run](#13-how-to-run)
14. [Future Scope](#14-future-scope)

---

## 1. Project Overview

### 1.1 Problem Statement

Air pollution is a critical concern in smart cities, with pollutants such as PM2.5, PM10, NO₂, SO₂, CO, and O₃ directly impacting public health. Traditional air quality monitoring systems provide only current readings without predictive capabilities, leaving residents unprepared for upcoming pollution spikes.

### 1.2 Objective

This project develops a **software-based IoT simulation system** for predicting Air Quality Index (AQI) and assessing health risks. The system:

- Collects air pollution data from publicly available datasets and live satellite APIs
- Applies a **Hybrid LSTM neural network** to forecast AQI levels for the next **24–72 hours**
- Uses **Random Forest classification** to categorize health risk levels (Good, Moderate, Unhealthy, Hazardous)
- Simulates an **IoT environment** by streaming real-time pollution and weather data
- Provides **alerts and health recommendations** based on predicted AQI values
- Stores all readings in a **MongoDB database** for historical analysis

### 1.3 Key Features

| Feature | Description |
|---------|-------------|
| **AQI Forecasting** | LSTM-based 24/48/72-hour multihorizon prediction |
| **Health Classification** | ML-powered risk categorization (Good → Hazardous) |
| **Live Data Integration** | Real-time satellite AQI + weather data from Open-Meteo API |
| **Interactive Dashboard** | 9-page dark-themed web dashboard with charts and alerts |
| **IoT Simulation** | Automated telemetry polling and logging system |
| **Database Storage** | MongoDB persistence for readings, predictions, and weather |
| **Alerting System** | Threshold-based AQI alerts with browser notifications |
| **Model Comparison** | LSTM vs ARIMA baseline performance analysis |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                             │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ data.csv     │  │ city_day.csv │  │ Open-Meteo API    │  │
│  │ (1990-2015)  │  │ (2015-2020)  │  │ (Live Satellite)  │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘  │
└─────────┼─────────────────┼───────────────────┼─────────────┘
          │                 │                   │
          ▼                 ▼                   │
   ┌──────────────────────────┐                 │
   │   merge_datasets.py      │                 │
   │   → master_dataset.csv   │                 │
   │   (335,292 records)      │                 │
   └──────────┬───────────────┘                 │
              │                                 │
              ▼                                 ▼
   ┌────────────────────┐            ┌──────────────────────┐
   │  TRAINING PIPELINE │            │   FastAPI Backend     │
   │                    │            │   (main.py)           │
   │  train_model.py    │──────────▶│   15 API Endpoints    │
   │  train_classifi... │    models/ │   Live AQI + Weather  │
   │  train_arima_...   │            │   LSTM Prediction     │
   └────────────────────┘            │   Health Classification│
                                     └──────────┬───────────┘
                                                │
                        ┌───────────────────────┼──────────────┐
                        │                       │              │
                        ▼                       ▼              ▼
              ┌─────────────────┐    ┌──────────────┐  ┌─────────────┐
              │  HTML Dashboard │    │   Streamlit   │  │   MongoDB   │
              │  (frontend/)    │    │   (app.py)    │  │  (atmosiq)  │
              │  9 pages        │    │   4 tabs      │  │  3 collections│
              └─────────────────┘    └──────────────┘  └─────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  IoT Simulator  │
              │  (iot_simulator)│
              │  → CSV logs     │
              └─────────────────┘
```

### 2.1 Data Flow

1. **Data Collection**: Historical CSV datasets (1990–2020) + live satellite API
2. **Data Preprocessing**: Column alignment, date formatting, AQI proxy calculation, dataset merge
3. **Model Training**: LSTM for forecasting, Random Forest for classification, ARIMA for baseline
4. **Backend API**: FastAPI serves predictions, live data, weather, and database endpoints
5. **Frontend**: Renders real-time dashboards with charts, gauges, and alerts
6. **Database**: MongoDB stores every AQI reading, prediction, and weather snapshot
7. **IoT Simulation**: Automated polling script mimics sensor network behavior

---

## 3. Technology Stack

### 3.1 Programming Languages
- **Python 3.13** — Backend, ML training, data processing
- **JavaScript (ES6+)** — Frontend dashboard interactivity
- **HTML5 / CSS3** — Dashboard structure and styling

### 3.2 Machine Learning
| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow | 2.13.1 | Deep learning framework |
| Keras | 3.10.0 | LSTM model construction |
| scikit-learn | 1.6.1 | Random Forest, Logistic Regression, preprocessing |
| statsmodels | 0.14.4 | ARIMA baseline model |
| NumPy | 1.24.3 | Numerical computation |
| Pandas | 2.3.3 | Data manipulation |

### 3.3 Backend
| Library | Version | Purpose |
|---------|---------|---------|
| FastAPI | 0.128.8 | REST API framework |
| Uvicorn | 0.39.0 | ASGI server |
| Requests | 2.32.5 | HTTP client for external APIs |
| joblib | 1.5.3 | Model serialization |

### 3.4 Frontend
| Technology | Purpose |
|------------|---------|
| Tailwind CSS (CDN) | Utility-first CSS framework |
| Chart.js | Interactive charts and visualizations |
| Material Symbols | Icon system |
| Google Fonts (Inter) | Typography |

### 3.5 Database & Testing
| Technology | Purpose |
|------------|---------|
| MongoDB 7.x | NoSQL document database |
| PyMongo 4.16 | Python MongoDB driver |
| Pytest 9.0.2 | Unit testing framework |
| Streamlit 1.12 | Alternative dashboard interface |

### 3.6 External APIs
| API | URL | Data Provided |
|-----|-----|---------------|
| Open-Meteo Air Quality | `air-quality-api.open-meteo.com` | PM2.5, PM10, NO₂, SO₂, CO, O₃, US AQI |
| Open-Meteo Weather | `api.open-meteo.com` | Temperature, Humidity, Wind Speed |

---

## 4. Project Structure

```
Hybrid-LSTM-AQI-Forcasting/
│
├── data/                           # Datasets
│   ├── data.csv                    # Historical pollution data (1990–2015, 62 MB)
│   ├── city_day.csv                # Modern CPCB dataset (2015–2020, 2.6 MB)
│   └── master_dataset.csv          # Merged master dataset (335,292 records, 14 MB)
│
├── models/                         # Trained ML model artifacts
│   ├── lstm_model.keras            # Trained LSTM neural network (807 KB)
│   ├── scaler.joblib               # MinMaxScaler for AQI normalization (975 B)
│   ├── health_classifier.joblib    # Random Forest classifier (140 MB)
│   ├── lstm_training_metrics.json  # LSTM evaluation metrics
│   ├── health_classifier_metrics.json  # Classifier evaluation metrics
│   └── arima_baseline_metrics.json # ARIMA baseline evaluation metrics
│
├── frontend/                       # Web dashboard
│   ├── index.html                  # Main dashboard (9 pages, 740+ lines)
│   └── app.js                      # Frontend logic (660+ lines)
│
├── tests/                          # Unit tests
│   └── test_api.py                 # 20 tests across 3 test classes
│
├── reports/                        # Generated reports
│   └── iot_simulation_log.csv      # IoT simulation output
│
├── main.py                         # FastAPI backend (690+ lines, 15 endpoints)
├── app.py                          # Streamlit dashboard (580+ lines, 4 tabs)
├── database.py                     # MongoDB database layer (190+ lines)
├── merge_datasets.py               # Data preprocessing & merge script
├── train_model.py                  # LSTM model training script
├── train_classification.py         # Health classifier training script
├── train_arima_baseline.py         # ARIMA baseline training script
├── iot_simulator.py                # IoT telemetry simulation script
├── requirements.txt                # Python dependencies
└── README.md                       # Project readme
```

---

## 5. Data Pipeline

### 5.1 Datasets

#### Historical Dataset (`data/data.csv`)
- **Source**: Open Government Data of India
- **Period**: 1990–2015
- **Records**: ~300,000
- **Key Columns**: `location`, `date`, `so2`, `no2`, `rspm` (PM10 proxy), `pm2_5`
- **Limitations**: No AQI column (AQI standard didn't exist pre-2015)

#### Modern Dataset (`data/city_day.csv`)
- **Source**: Central Pollution Control Board (CPCB) via Kaggle
- **Period**: 2015–2020
- **Records**: ~30,000
- **Key Columns**: `City`, `Date`, `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`, `AQI`, `AQI_Bucket`

### 5.2 Data Preprocessing (`merge_datasets.py`)

The merge script performs the following operations:

1. **Column Standardization**: Maps historical column names to modern format
   - `location` → `City`, `date` → `Date`, `rspm` → `PM10`, `no2` → `NO2`

2. **Date Normalization**: Converts all dates to `datetime` format

3. **AQI Proxy Calculation**: For historical records (pre-2015):
   ```
   AQI = max(PM10, NO2) × 1.5
   ```
   This provides a rough algorithmic proxy based on the dominant pollutants.

4. **Dataset Fusion**: Concatenates both datasets, sorts by City + Date

5. **Deduplication**: Keeps the latest record when both datasets overlap

6. **Output**: `data/master_dataset.csv` — **335,292 records** with columns:
   `City, Date, PM2.5, PM10, NO2, SO2, CO, O3, AQI, AQI_Bucket`

### 5.3 Live Data Fetching

The system fetches real-time data from two free, keyless APIs:

```python
# AQI Data (6 pollutants + US AQI)
GET https://air-quality-api.open-meteo.com/v1/air-quality
    ?latitude=28.61&longitude=77.20
    &current=pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,us_aqi

# Weather Data (3 parameters)
GET https://api.open-meteo.com/v1/forecast
    ?latitude=28.61&longitude=77.20
    &current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code
```

**Supported Cities** (8 Indian cities with coordinates):
Delhi, Mumbai, Bangalore, Kolkata, Chennai, Hyderabad, Ahmedabad, Jaipur

---

## 6. Machine Learning Models

### 6.1 LSTM Forecasting Model (`train_model.py`)

#### Purpose
Predicts AQI values for the next 24, 48, and 72 hours using a 30-day historical sequence.

#### Architecture
```
Input Layer: (30 timesteps × 1 feature)
    ↓
LSTM Layer 1: 64 units, return_sequences=True
    ↓
Dropout: 0.2
    ↓
LSTM Layer 2: 64 units, return_sequences=True
    ↓
Dropout: 0.2
    ↓
LSTM Layer 3: 32 units, return_sequences=False
    ↓
Dropout: 0.2
    ↓
Dense Layer: 16 units, activation='relu'
    ↓
Output Dense: 1 unit (predicted AQI)
```

#### Training Details
| Parameter | Value |
|-----------|-------|
| Sequence Length | 30 days |
| Feature | AQI (univariate, MinMaxScaler normalized) |
| Train/Test Split | 80/20 |
| Optimizer | Adam |
| Loss Function | Mean Squared Error |
| Batch Size | 64 |
| Max Epochs | 50 |
| Early Stopping | Patience=5, restores best weights |
| Validation Split | 10% of training data |
| Training Sequences | 152,576 |
| Test Sequences | 38,144 |
| Cities Covered | 243 |

#### Multi-Horizon Forecasting
The system uses **iterative prediction** to forecast 24h, 48h, and 72h:

```python
# Predict next day, then feed prediction back as input
for horizon in [24, 48, 72]:
    pred = model.predict(current_sequence)
    current_sequence = np.append(current_sequence[1:], pred)
```

A **mean-reversion mechanism** prevents drift in multi-step predictions:
```python
reversion_factor = 0.15 * (step / total_steps)
pred = pred * (1 - reversion_factor) + historical_mean * reversion_factor
```

#### Model Artifacts
- `models/lstm_model.keras` (807 KB) — Trained neural network
- `models/scaler.joblib` (975 B) — MinMaxScaler for AQI normalization

---

### 6.2 Health Risk Classifier (`train_classification.py`)

#### Purpose
Classifies health risk levels based on pollutant concentrations using traditional ML models.

#### Models Trained

**1. Random Forest Classifier** (Selected as best model)
| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest |
| Estimators | 120 trees |
| Max Depth | None (unlimited) |
| Class Weights | Balanced |
| Features | PM2.5, PM10, NO₂, SO₂, City (label-encoded) |
| Target | 4 classes: Good, Moderate, Unhealthy, Hazardous |

**2. Logistic Regression** (Baseline comparison)
| Parameter | Value |
|-----------|-------|
| Solver | lbfgs |
| Max Iterations | 1000 |
| Penalty | L2 |

#### Classification Pipeline
```
Raw Data → Impute Missing Values (median) → Standard Scaling → Model
```

#### AQI-to-Health Risk Mapping
| AQI Range | Health Category |
|-----------|----------------|
| 0–50 | Good |
| 51–100 | Moderate |
| 101–200 | Unhealthy |
| 201+ | Hazardous |

#### Model Artifact
- `models/health_classifier.joblib` (140 MB) — Trained Random Forest pipeline

---

### 6.3 ARIMA Baseline (`train_arima_baseline.py`)

#### Purpose
Provides a traditional statistical baseline for comparison with the LSTM model.

#### Configuration
| Parameter | Value |
|-----------|-------|
| Model | ARIMA |
| Order | (2, 1, 2) — AR=2, Differencing=1, MA=2 |
| Holdout | Last 72 days per city |
| Cities Trained | 8 (Delhi, Mumbai, Chennai, Hyderabad, etc.) |

---

## 7. Backend API (FastAPI)

### 7.1 Server Configuration
- **Framework**: FastAPI with Uvicorn ASGI server
- **Port**: 8000
- **CORS**: Enabled for all origins (development mode)
- **Static Files**: Serves `frontend/` directory

### 7.2 API Endpoints (15 Total)

#### Core Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check — returns server status |
| GET | `/api/cities` | Returns list of 8 supported cities |

#### AQI Data
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/current-status?city=Delhi` | Live AQI + weather data (satellite first, CSV fallback) |
| POST | `/api/predict?city=Delhi` | LSTM 24/48/72h AQI prediction |
| GET | `/api/historical?city=Delhi` | Historical AQI trend data |
| GET | `/api/compare?cities=Delhi,Mumbai,...` | Multi-city AQI comparison |

#### Weather
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/weather?city=Delhi` | Current temperature, humidity, wind speed |

#### Model & Metrics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/model-metrics` | LSTM + Classifier + ARIMA evaluation metrics |
| GET | `/api/classifier-status` | Health classifier availability check |

#### Reports & Export
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/report-summary?city=Delhi` | Full JSON report with all data |
| POST | `/api/report-summary/export?city=Delhi` | Save report to disk |
| GET | `/api/export-csv?city=Delhi` | Download historical data as CSV |

#### Database
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/db-status` | MongoDB connection status + collection counts |
| GET | `/api/readings?city=Delhi&limit=50` | Stored AQI readings from MongoDB |
| GET | `/api/stored-predictions?city=Delhi` | Stored LSTM predictions from MongoDB |

#### Frontend
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/app` | Serves the HTML dashboard |

### 7.3 Data Source Priority
```
1. Live Satellite API (Open-Meteo)  →  Primary source
2. CSV Dataset (master_dataset.csv) →  Automatic fallback if API is unreachable
```

### 7.4 Keras Compatibility Patch
The backend includes a compatibility patch for loading Keras models across different versions:
```python
# Fixes "Could not locate class 'Orthogonal'" errors
keras.config.enable_unsafe_deserialization()
```

---

## 8. Frontend Dashboard

### 8.1 HTML Dashboard (`frontend/index.html` + `app.js`)

A **dark-themed, single-page application** with 9 navigable pages:

#### Page 1: Dashboard (System Overview)
- AQI circular gauge with real-time value and color coding
- Pollutant concentration cards (PM2.5, PM10, NO₂, SO₂)
- Health impact badge and data source indicator
- Risk classification panel
- **Weather strip**: Temperature (°C), Humidity (%), Wind Speed (km/h)

#### Page 2: Live Monitor (Simulated IoT Feed)
- Real-time pollutant bars with status tags (OPTIMAL / NORMAL / ELEVATED)
- **6 pollutant cards**: PM2.5, PM10, NO₂, SO₂, CO, O₃ — each with animated progress bars
- API Response Log (timestamped telemetry stream)
- Weather telemetry cards (Temperature, Humidity, Wind Speed)

#### Page 3: Forecast (AI Prediction)
- "Run AI Prediction" button triggers LSTM inference
- 3 prediction cards: 24h, 48h, 72h forecast with AQI value and health risk
- Loading animation during inference

#### Page 4: Health Risk Assessment
- Current health risk level with color-coded display
- Health advice text based on AQI level
- Precautions checklist (toggleable safety recommendations)
- AQI scale reference card

#### Page 5: Historical Trends
- "Fetch Historical Data" button loads city-specific trends
- **AQI Trend Chart** (Chart.js line plot)
- **Pollutant Breakdown Chart** (multi-line chart for PM2.5, PM10, NO₂, SO₂)
- Data table with sortable columns

#### Page 6: City Comparison
- Compares AQI across Delhi, Mumbai, Bangalore, Kolkata, Chennai
- AQI comparison bar chart
- Individual city cards with current values

#### Page 7: Model Metrics
- LSTM model metrics (MAE, RMSE, epochs, architecture)
- Health classifier metrics (accuracy, F1-score, per-class precision/recall)
- ARIMA baseline metrics (per-city MAE/RMSE comparison)

#### Page 8: Alerts
- Custom AQI threshold setting (stored in localStorage)
- Alert history log with timestamps
- Test alert button for debugging
- Browser notification permission management

#### Page 9: Reports & Export
- Generate full JSON report
- Download historical data as CSV
- Report viewer panel

### 8.2 Design System
| Design Element | Implementation |
|----------------|----------------|
| Color Scheme | Dark mode with neon green (#57fe81) primary, red (#ff716a) secondary |
| Typography | Inter font family from Google Fonts |
| Layout | CSS Grid (12-column) with glassmorphism effects |
| Animations | CSS pulse-glow, smooth transitions, progress bar fills |
| Icons | Material Symbols (filled variant) |
| Responsiveness | Fixed 256px sidebar + fluid main content area |

### 8.3 Sidebar Status Indicators
- **API Connected** — Green dot when FastAPI is reachable
- **MongoDB Connected** — Green dot when MongoDB is active
- Auto-refreshes every 60 seconds

### 8.4 Streamlit Dashboard (`app.py`)

An alternative dashboard built with Streamlit, featuring **4 tabs**:

| Tab | Content |
|-----|---------|
| 📊 Current Data | AQI overview, CO/O3/SO₂ metrics, weather conditions, last 5 days table |
| 🔮 Prediction (AI) | LSTM forecast with line chart, health risk assessment |
| 📈 Historical Trends | Interactive Plotly charts for AQI and pollutant trends |
| 🌡️ Weather & IoT | Weather conditions, MongoDB status, stored readings/predictions tables |

**Sidebar features**: City selection, API toggle, report export, MongoDB connection status

---

## 9. Database Layer (MongoDB)

### 9.1 Configuration
| Parameter | Value |
|-----------|-------|
| Database Engine | MongoDB |
| Connection URI | `mongodb://localhost:27017/` |
| Database Name | `atmosiq` |
| Connection Timeout | 3000ms |

### 9.2 Collections

#### `aqi_readings` — AQI Observation Snapshots
```json
{
  "city": "Delhi",
  "source": "LIVE SATELLITE",
  "aqi": 198,
  "pm25": 73.1,
  "pm10": 387.4,
  "no2": 26.3,
  "so2": 34.0,
  "temperature": 38.2,
  "humidity": 22,
  "wind_speed": 8.5,
  "timestamp": "2026-04-07T05:04:07Z"
}
```

#### `predictions` — LSTM Forecast Results
```json
{
  "city": "Delhi",
  "data_source": "LIVE",
  "pred_24h": 185.42,
  "pred_48h": 172.18,
  "pred_72h": 165.90,
  "timestamp": "2026-04-07T05:04:09Z"
}
```

#### `weather_logs` — Weather Telemetry
```json
{
  "city": "Delhi",
  "temperature": 38.2,
  "humidity": 22,
  "wind_speed": 8.5,
  "timestamp": "2026-04-07T05:04:07Z"
}
```

### 9.3 Graceful Fallback
The `database.py` module is designed to be **fault-tolerant**:
- If MongoDB is unreachable, the system continues in CSV-only mode
- No crashes or errors propagate to the API or frontend
- The `/api/db-status` endpoint reports `"connected": false` with an explanatory message

### 9.4 API Functions
| Function | Description |
|----------|-------------|
| `get_db()` | Returns MongoDB handle or None |
| `is_connected()` | Check connection status |
| `store_reading()` | Save AQI observation |
| `get_readings()` | Query recent readings |
| `store_prediction()` | Save LSTM forecast |
| `get_predictions()` | Query stored predictions |
| `store_weather()` | Save weather snapshot |
| `get_db_stats()` | Return collection counts |

---

## 10. IoT Simulation

### 10.1 Purpose
The `iot_simulator.py` script simulates an IoT sensor network by periodically polling the backend API and logging telemetry data, mimicking how real-world IoT devices would interact with the system.

### 10.2 Simulation Parameters
| Parameter | Default | CLI Argument |
|-----------|---------|-------------|
| Base URL | `http://127.0.0.1:8000` | `--base-url` |
| Cities | Delhi, Mumbai, Bangalore, Kolkata, Chennai | `--cities` |
| Interval | 120 seconds | `--interval` |
| Iterations | 5 cycles | `--iterations` |

### 10.3 Telemetry Log Format
Each simulation cycle produces a CSV row per city:
```
timestamp, city, source, aqi_now, pm25, pm10, no2, so2,
temperature, humidity, wind_speed, risk_now, aqi_24h, aqi_48h, aqi_72h
```

### 10.4 Output
- Console: Real-time status with AQI, temperature, and prediction values
- File: `reports/iot_simulation_log.csv`

### 10.5 Usage
```bash
python iot_simulator.py --iterations 10 --interval 60
```

---

## 11. Testing

### 11.1 Test Suite (`tests/test_api.py`)

**20 unit tests** organized into 3 test classes:

#### TestClassifyAqi (9 tests)
Tests the rule-based AQI health risk classification logic:
- Tests all 6 AQI levels (Good, Moderate, Unhealthy for Sensitive, Unhealthy, Very Unhealthy, Hazardous)
- Tests boundary values (AQI=50, 100, 301)
- Validates color codes and level strings

#### TestDatabaseModule (4 tests)
Tests the MongoDB database helper module:
- Module import and function availability
- `get_db_stats()` returns valid dictionary
- `get_readings()` returns list
- `get_predictions()` returns list

#### TestAPIEndpoints (7 tests)
Tests FastAPI routes using httpx TestClient:
- `GET /` — Health check
- `GET /api/cities` — City list (validates Delhi is present)
- `GET /api/db-status` — MongoDB status
- `GET /api/weather` — Weather endpoint
- `GET /api/model-metrics` — All model metrics
- `GET /api/classifier-status` — Classifier status
- `GET /api/readings` — Stored readings

### 11.2 Running Tests
```bash
python -m pytest tests/ -v
```

### 11.3 Test Results
```
tests/test_api.py::TestClassifyAqi::test_good_range PASSED
tests/test_api.py::TestClassifyAqi::test_moderate_range PASSED
tests/test_api.py::TestClassifyAqi::test_unhealthy_sensitive PASSED
tests/test_api.py::TestClassifyAqi::test_unhealthy PASSED
tests/test_api.py::TestClassifyAqi::test_very_unhealthy PASSED
tests/test_api.py::TestClassifyAqi::test_hazardous PASSED
tests/test_api.py::TestClassifyAqi::test_boundary_50 PASSED
tests/test_api.py::TestClassifyAqi::test_boundary_100 PASSED
tests/test_api.py::TestClassifyAqi::test_boundary_301 PASSED
tests/test_api.py::TestDatabaseModule::test_import_database PASSED
tests/test_api.py::TestDatabaseModule::test_db_stats_returns_dict PASSED
tests/test_api.py::TestDatabaseModule::test_get_readings_returns_list PASSED
tests/test_api.py::TestDatabaseModule::test_get_predictions_returns_list PASSED
tests/test_api.py::TestAPIEndpoints::test_root_endpoint PASSED
tests/test_api.py::TestAPIEndpoints::test_cities_endpoint PASSED
tests/test_api.py::TestAPIEndpoints::test_db_status_endpoint PASSED
tests/test_api.py::TestAPIEndpoints::test_weather_endpoint PASSED
tests/test_api.py::TestAPIEndpoints::test_model_metrics_endpoint PASSED
tests/test_api.py::TestAPIEndpoints::test_classifier_status_endpoint PASSED
tests/test_api.py::TestAPIEndpoints::test_readings_endpoint PASSED
====================== 20 passed in 6.93s ======================
```

---

## 12. Results and Evaluation

### 12.1 LSTM Forecasting Performance

| Metric | Value |
|--------|-------|
| **MAE** (Mean Absolute Error) | **22.47** AQI points |
| **RMSE** (Root Mean Square Error) | **31.85** AQI points |
| Best Validation Loss | 0.0018 |
| Epochs Trained | 23 (of 50 max, early stopped) |
| Training Sequences | 152,576 |
| Test Sequences | 38,144 |

### 12.2 Health Classifier Performance

#### Random Forest (Selected Model)
| Metric | Value |
|--------|-------|
| **Accuracy** | **97.57%** |
| **F1-Score (Weighted)** | **0.9757** |

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Good | 0.961 | 0.963 | 0.962 | 2,691 |
| Moderate | 0.971 | 0.975 | 0.973 | 6,161 |
| Unhealthy | 0.974 | 0.976 | 0.975 | 8,730 |
| Hazardous | 0.990 | 0.981 | 0.986 | 6,419 |

#### Logistic Regression (Baseline)
| Metric | Value |
|--------|-------|
| Accuracy | 76.61% |
| F1-Score (Weighted) | 0.7730 |

### 12.3 ARIMA Baseline Performance

| Metric | Value |
|--------|-------|
| **MAE (Mean)** | **26.15** AQI points |
| **RMSE (Mean)** | **33.01** AQI points |
| Cities Trained | 8 |

#### Per-City ARIMA Results
| City | MAE | RMSE |
|------|-----|------|
| Nagpur | 14.57 | 28.19 |
| Chennai | 17.79 | 25.58 |
| Hyderabad | 18.12 | 22.21 |
| Jaipur | 18.93 | 23.92 |
| Chandigarh | 20.17 | 25.76 |
| Delhi | 28.31 | 39.50 |
| Lucknow | 38.01 | 42.29 |
| Ahmedabad | 53.34 | 56.61 |

### 12.4 Model Comparison: LSTM vs ARIMA

| Metric | LSTM | ARIMA | LSTM Improvement |
|--------|------|-------|------------------|
| MAE | **22.47** | 26.15 | **14.1% better** |
| RMSE | **31.85** | 33.01 | **3.5% better** |

**Conclusion**: The LSTM model outperforms the traditional ARIMA baseline on both metrics, demonstrating the effectiveness of deep learning for AQI time-series forecasting.

---

## 13. How to Run

### 13.1 Prerequisites
- Python 3.10+
- MongoDB Server running locally (port 27017)
- Internet connection (for live API data)

### 13.2 Installation
```bash
# Clone the repository
git clone https://github.com/Raaja-Santhosh/Hybrid-LSTM-AQI-Forcasting.git
cd Hybrid-LSTM-AQI-Forcasting

# Install dependencies
pip install -r requirements.txt
```

### 13.3 Data Preparation (First-time Only)
```bash
# Merge datasets into master_dataset.csv
python merge_datasets.py
```

### 13.4 Model Training (If Models Not Present)
```bash
# Train LSTM forecasting model (~15 min)
python train_model.py

# Train health risk classifier (~5 min)
python train_classification.py

# Train ARIMA baseline (~2 min)
python train_arima_baseline.py
```

### 13.5 Start the Application
```bash
# Start FastAPI backend
uvicorn main:app --reload --port 8000

# Open in browser
# → http://127.0.0.1:8000/app (HTML Dashboard)
```

### 13.6 Alternative: Streamlit Dashboard
```bash
# Start Streamlit (separate terminal)
streamlit run app.py
```

### 13.7 Run IoT Simulation
```bash
# Ensure backend is running first, then:
python iot_simulator.py --iterations 10 --interval 60
```

### 13.8 Run Tests
```bash
python -m pytest tests/ -v
```

---

## 14. Future Scope

1. **Cloud Deployment**: Deploy to AWS/GCP/Azure for public access with auto-scaling
2. **Real IoT Hardware**: Integrate with physical sensors (MQ-135, DSM501A) via MQTT
3. **Multivariate LSTM**: Include weather features (temperature, humidity, wind) as additional model inputs
4. **User Authentication**: Add login system for personalized thresholds and saved cities
5. **Mobile App**: Build a React Native / Flutter companion app with push notifications
6. **Hourly Granularity**: Shift from daily to hourly predictions for more precise forecasting
7. **GIS Mapping**: Add geospatial AQI heatmaps using Leaflet.js or Mapbox
8. **Ensemble Models**: Combine LSTM + ARIMA + XGBoost for more robust predictions

---

## Team Contributions

| Role | Responsibility |
|------|---------------|
| **Data Preprocessing & Feature Engineering** | Dataset merging, AQI proxy calculation, scaler fitting, CO/O3 integration |
| **Time-Series Model Development (LSTM)** | Neural network architecture design, training pipeline, multi-horizon forecasting |
| **Classification Model Development** | Random Forest + Logistic Regression training, evaluation, auto-selection |
| **Backend API Integration** | FastAPI server, 15 endpoints, database integration, live API proxying |
| **Frontend Dashboard & Visualization** | HTML/JS dashboard (9 pages), Streamlit app (4 tabs), Chart.js graphs, alerts |

---

*Document generated: April 2026*
*Project: Hybrid LSTM-Based AQI Forecasting and Health Risk Classification*
