import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib
import sys
import json
import requests

st.set_page_config(page_title="AQI Prediction System", layout="wide")

st.title("🌫 Hybrid AQI Forecasting and Health Risk Classification System")
st.markdown("Predict the air quality and health risk index based on historical pollution data.")

# Streamlit < 1.18 does not have cache_data.
cache_data = getattr(st, "cache_data", st.cache)
if hasattr(st, "cache_resource"):
    cache_resource = st.cache_resource
elif hasattr(st, "experimental_singleton"):
    cache_resource = st.experimental_singleton
else:
    def cache_resource(func):
        return func

SEQUENCE_LENGTH = 30
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


def classify_aqi_risk(aqi_value):
    if aqi_value <= 50:
        return "Good"
    if aqi_value <= 100:
        return "Satisfactory"
    if aqi_value <= 200:
        return "Moderate"
    if aqi_value <= 300:
        return "Poor"
    if aqi_value <= 400:
        return "Very Poor"
    return "Severe"


def get_risk_recommendation(risk_level):
    recommendations = {
        "Good": "Air quality is favorable. Outdoor activities are safe.",
        "Satisfactory": "Air quality is acceptable. Sensitive groups should monitor prolonged exposure.",
        "Moderate": "Limit heavy outdoor exertion, especially for children and elderly.",
        "Poor": "Wear a mask outdoors and avoid long exposure in traffic-heavy areas.",
        "Very Poor": "Avoid outdoor exercise. Use air purifiers indoors if possible.",
        "Severe": "Health emergency conditions. Stay indoors and minimize all outdoor activity.",
    }
    return recommendations.get(risk_level, "Monitor AQI closely and follow health advisories.")


def map_classifier_label(label):
    mapping = {
        "Good": "Good",
        "Moderate": "Satisfactory",
        "Unhealthy": "Poor",
        "Hazardous": "Severe",
    }
    return mapping.get(str(label), str(label))


def forecast_multi_horizon(model, scaler, latest_sequence, steps):
    """Iteratively predicts future AQI values using last predictions as next inputs."""
    seq = latest_sequence.astype(float).copy().reshape(-1)
    predictions = []

    for _ in range(steps):
        scaled_seq = scaler.transform(seq.reshape(-1, 1))
        model_input = np.reshape(scaled_seq, (1, len(seq), 1))
        pred_scaled = model.predict(model_input, verbose=0)
        pred = float(scaler.inverse_transform(pred_scaled)[0][0])
        predictions.append(pred)
        seq = np.append(seq[1:], pred)

    return predictions


@cache_resource
def load_artifacts():
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
        return None, None, f"Failed to load model artifacts: {exc}"


@cache_resource
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
        return None, f"Failed to load health classifier: {exc}"


@cache_data
def load_health_classifier_metrics():
    metrics_path = "models/health_classifier_metrics.json"
    if not os.path.exists(metrics_path):
        return None, "Classifier metrics file not found in models/"

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as exc:
        return None, f"Failed to load classifier metrics: {exc}"


@cache_data
def load_lstm_metrics():
    metrics_path = "models/lstm_training_metrics.json"
    if not os.path.exists(metrics_path):
        return None, "LSTM metrics file not found in models/"

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as exc:
        return None, f"Failed to load LSTM metrics: {exc}"


@cache_data
def load_arima_metrics():
    metrics_path = "models/arima_baseline_metrics.json"
    if not os.path.exists(metrics_path):
        return None, "ARIMA metrics file not found in models/"

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as exc:
        return None, f"Failed to load ARIMA metrics: {exc}"


def predict_health_risk_ml(artifact, city, latest_record):
    if artifact is None:
        return None

    try:
        feature_cols = artifact["features"]
        row = {}
        for col in feature_cols:
            if col == "City":
                row[col] = city
            else:
                row[col] = latest_record.get(col, np.nan)

        x_input = pd.DataFrame([row])
        pred = artifact["model"].predict(x_input)[0]
        return map_classifier_label(pred)
    except Exception:
        return None


def fetch_api_json(base_url, endpoint, method="get", params=None):
    url = base_url.rstrip("/") + endpoint
    try:
        if method.lower() == "post":
            response = requests.post(url, params=params, timeout=15)
        else:
            response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        return None, str(exc)

# --- LOAD DATA ---
@cache_data
def load_data():
    file_path = "data/city_day.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Basic preprocessing for the prototype
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

df = load_data()
model, scaler, model_error = load_artifacts()
health_classifier, health_classifier_error = load_health_classifier_artifact()
classifier_metrics, classifier_metrics_error = load_health_classifier_metrics()
lstm_metrics, lstm_metrics_error = load_lstm_metrics()
arima_metrics, arima_metrics_error = load_arima_metrics()

# --- SIDEBAR ---
st.sidebar.header("🔧 Settings")
if df is not None:
    cities = df['City'].unique()
    selected_city = st.sidebar.selectbox("Select City", cities)
    use_backend_api = st.sidebar.checkbox("Use Backend API", value=True)
    api_base_url = st.sidebar.text_input("Backend API URL", value=DEFAULT_API_BASE_URL)
    
    city_data = df[df['City'] == selected_city].dropna(subset=['AQI'])
    
    st.sidebar.markdown("---")
    st.sidebar.success("Data Loaded Successfully!")
else:
    st.sidebar.error("Error: `data/city_day.csv` not found!")

# --- MAIN DASHBOARD ---
tab1, tab2, tab3 = st.tabs(["📊 Current Data", "🔮 Prediction (AI)", "📈 Historical Trends"])

api_current, api_current_error = (None, None)
api_predict, api_predict_error = (None, None)
api_metrics, api_metrics_error = (None, None)

if df is not None and use_backend_api:
    api_current, api_current_error = fetch_api_json(api_base_url, "/api/current-status", params={"city": selected_city})
    api_predict, api_predict_error = fetch_api_json(api_base_url, "/api/predict", method="post", params={"city": selected_city})
    api_metrics, api_metrics_error = fetch_api_json(api_base_url, "/api/model-metrics")

with st.sidebar:
    st.markdown("---")
    st.subheader("Reports")
    if st.button("Export Summary Report"):
        if use_backend_api:
            export_data, export_error = fetch_api_json(
                api_base_url,
                "/api/report-summary/export",
                method="post",
                params={"city": selected_city},
            )
            if export_error:
                st.error(f"Export failed: {export_error}")
            else:
                st.success(f"Report saved: {export_data.get('file', 'unknown path')}")
        else:
            st.info("Enable 'Use Backend API' to export report summary.")

with tab1:
    st.header(f"Current AQI Overview: {selected_city if df is not None else 'N/A'}")
    if use_backend_api and api_current is not None and "error" not in api_current:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AQI", api_current.get("aqi", "N/A"))
        col2.metric("PM2.5", api_current.get("pm25", "N/A"))
        col3.metric("NO2", api_current.get("no2", "N/A"))
        risk_level = api_current.get("health_risk", {}).get("level", "Unknown")
        col4.metric("Risk Level", risk_level)

        ml_risk = api_current.get("health_risk_ml", {})
        if isinstance(ml_risk, dict) and ml_risk.get("level"):
            st.info(f"ML Health Classification: {map_classifier_label(ml_risk.get('level'))}")

        if df is not None and not city_data.empty:
            st.subheader("Recent Data (Last 5 Days)")
            st.dataframe(city_data.tail(5)[['Date', 'PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI', 'AQI_Bucket']])
    elif df is not None and not city_data.empty:
        latest_record = city_data.iloc[-1]
        ml_health_risk = predict_health_risk_ml(health_classifier, selected_city, latest_record)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AQI", latest_record['AQI'])
        col2.metric("PM2.5", latest_record['PM2.5'])
        col3.metric("NO2", latest_record['NO2'])
        col4.metric("Risk Level", latest_record.get('AQI_Bucket', 'Unknown'))

        if ml_health_risk:
            st.info(f"ML Health Classification: {ml_health_risk}")
        elif health_classifier_error:
            st.caption(f"Classifier status: {health_classifier_error}")
        
        st.subheader("Recent Data (Last 5 Days)")
        st.dataframe(city_data.tail(5)[['Date', 'PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI', 'AQI_Bucket']])
    else:
        if use_backend_api and api_current_error:
            st.warning(f"Backend API unavailable, using local data. Error: {api_current_error}")
        st.info("No data available to display.")

with tab2:
    st.header("Future AQI Prediction")
    if use_backend_api and api_predict is not None and "error" not in api_predict:
        preds = api_predict.get("predictions", {})
        pred24 = float(preds.get("24h", {}).get("aqi", api_predict.get("predicted_aqi_tomorrow", 0.0)))
        pred48 = float(preds.get("48h", {}).get("aqi", pred24))
        pred72 = float(preds.get("72h", {}).get("aqi", pred48))
        latest_aqi = float(city_data['AQI'].iloc[-1]) if df is not None and not city_data.empty else pred24
        delta = pred24 - latest_aqi

        st.success("Prediction served from backend API.")
        st.metric("Predicted AQI (24h)", f"{pred24:.2f}", delta=f"{delta:+.2f}")
        st.metric("Predicted Health Risk", classify_aqi_risk(pred24))

        col24, col48, col72 = st.columns(3)
        col24.metric("AQI +24h", f"{pred24:.2f}")
        col48.metric("AQI +48h", f"{pred48:.2f}")
        col72.metric("AQI +72h", f"{pred72:.2f}")

        forecast_df = pd.DataFrame(
            {
                "Horizon (hours)": [24, 48, 72],
                "Predicted AQI": [pred24, pred48, pred72],
                "Risk": [
                    classify_aqi_risk(pred24),
                    classify_aqi_risk(pred48),
                    classify_aqi_risk(pred72),
                ],
            }
        )
        st.dataframe(forecast_df, use_container_width=True)

        worst_horizon = max([(24, pred24), (48, pred48), (72, pred72)], key=lambda x: x[1])
        worst_risk = classify_aqi_risk(worst_horizon[1])
        st.warning(
            f"Alert Outlook: Highest forecasted risk is **{worst_risk}** at **+{worst_horizon[0]}h**. "
            f"{get_risk_recommendation(worst_risk)}"
        )
    elif model_error:
        st.error(model_error)
    elif df is None or city_data.empty:
        st.info("No city data available for prediction.")
    elif len(city_data) < SEQUENCE_LENGTH:
        st.warning(f"Need at least {SEQUENCE_LENGTH} AQI records for prediction.")
    else:
        latest_sequence = city_data['AQI'].tail(SEQUENCE_LENGTH).values.reshape(-1, 1)
        future_preds = forecast_multi_horizon(model, scaler, latest_sequence, steps=72)

        predicted_aqi = float(future_preds[23])
        latest_aqi = float(city_data['AQI'].iloc[-1])
        delta = predicted_aqi - latest_aqi

        st.success("Trained model loaded successfully.")
        st.metric("Predicted AQI (24h)", f"{predicted_aqi:.2f}", delta=f"{delta:+.2f}")
        st.metric("Predicted Health Risk", classify_aqi_risk(predicted_aqi))

        col24, col48, col72 = st.columns(3)
        col24.metric("AQI +24h", f"{future_preds[23]:.2f}")
        col48.metric("AQI +48h", f"{future_preds[47]:.2f}")
        col72.metric("AQI +72h", f"{future_preds[71]:.2f}")

        forecast_df = pd.DataFrame(
            {
                "Horizon (hours)": [24, 48, 72],
                "Predicted AQI": [future_preds[23], future_preds[47], future_preds[71]],
                "Risk": [
                    classify_aqi_risk(future_preds[23]),
                    classify_aqi_risk(future_preds[47]),
                    classify_aqi_risk(future_preds[71]),
                ],
            }
        )
        st.dataframe(forecast_df, use_container_width=True)

        worst_horizon = max([(24, future_preds[23]), (48, future_preds[47]), (72, future_preds[71])], key=lambda x: x[1])
        worst_risk = classify_aqi_risk(worst_horizon[1])
        st.warning(
            f"Alert Outlook: Highest forecasted risk is **{worst_risk}** at **+{worst_horizon[0]}h**. "
            f"{get_risk_recommendation(worst_risk)}"
        )

        if health_classifier is not None:
            ml_health_risk = predict_health_risk_ml(health_classifier, selected_city, city_data.iloc[-1])
            if ml_health_risk:
                st.metric("Predicted Health Risk (ML Classifier)", ml_health_risk)
        elif health_classifier_error:
            st.caption(f"Classifier status: {health_classifier_error}")

        if use_backend_api and api_predict_error:
            st.caption(f"Backend prediction fallback in use. API error: {api_predict_error}")

with tab3:
    st.header("Historical Pollution Trends")
    if df is not None and not city_data.empty:
        fig = px.line(city_data, x="Date", y="AQI", title=f"Historical AQI for {selected_city}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data to display.")

st.markdown("---")
st.subheader("Classifier Performance")
metrics_source = classifier_metrics
if use_backend_api and api_metrics is not None and isinstance(api_metrics, dict):
    metrics_source = api_metrics.get("classifier", {}).get("metrics", classifier_metrics)

if metrics_source is not None:
    selected_model = metrics_source.get("selected_model", "N/A")
    selected_f1 = metrics_source.get("selected_model_f1_weighted", None)
    all_metrics = metrics_source.get("all_metrics", {})

    c1, c2 = st.columns(2)
    c1.metric("Selected Classifier", str(selected_model))
    if selected_f1 is not None:
        c2.metric("Weighted F1", f"{float(selected_f1):.4f}")
    else:
        c2.metric("Weighted F1", "N/A")

    metrics_rows = []
    for model_name, values in all_metrics.items():
        metrics_rows.append(
            {
                "Model": model_name,
                "Accuracy": float(values.get("accuracy", 0.0)),
                "F1 (weighted)": float(values.get("f1_weighted", 0.0)),
            }
        )

    if metrics_rows:
        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)
else:
    err_msg = classifier_metrics_error or "Classifier metrics not available yet."
    if use_backend_api and api_metrics_error:
        err_msg += f" API error: {api_metrics_error}"
    st.caption(err_msg)

st.subheader("LSTM Training Metrics")
lstm_source = lstm_metrics
if use_backend_api and api_metrics is not None and isinstance(api_metrics, dict):
    lstm_source = api_metrics.get("lstm", {}).get("metrics", lstm_metrics)

if lstm_source is not None:
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{float(lstm_source.get('mae', 0.0)):.2f}")
    m2.metric("RMSE", f"{float(lstm_source.get('rmse', 0.0)):.2f}")
    m3.metric("Epochs Trained", str(lstm_source.get('epochs_ran', 'N/A')))

    st.dataframe(pd.DataFrame([lstm_source]), use_container_width=True)
else:
    err_msg = lstm_metrics_error or "LSTM metrics not available yet."
    if use_backend_api and api_metrics_error:
        err_msg += f" API error: {api_metrics_error}"
    st.caption(err_msg)

st.subheader("ARIMA Baseline Metrics")
arima_source = arima_metrics
if use_backend_api and api_metrics is not None and isinstance(api_metrics, dict):
    arima_source = api_metrics.get("arima_baseline", {}).get("metrics", arima_metrics)

if arima_source is not None:
    agg = arima_source.get("aggregate", {})
    a1, a2, a3 = st.columns(3)
    a1.metric("ARIMA MAE (mean)", f"{float(agg.get('mae_mean', 0.0)):.2f}")
    a2.metric("ARIMA RMSE (mean)", f"{float(agg.get('rmse_mean', 0.0)):.2f}")
    a3.metric("Cities Trained", str(arima_source.get("cities_trained", "N/A")))

    if lstm_source is not None:
        c1, c2 = st.columns(2)
        c1.metric("LSTM MAE", f"{float(lstm_source.get('mae', 0.0)):.2f}")
        c2.metric("LSTM RMSE", f"{float(lstm_source.get('rmse', 0.0)):.2f}")

    per_city = arima_source.get("per_city", {})
    if per_city:
        city_rows = []
        for city, vals in per_city.items():
            city_rows.append(
                {
                    "City": city,
                    "ARIMA MAE": float(vals.get("mae", 0.0)),
                    "ARIMA RMSE": float(vals.get("rmse", 0.0)),
                }
            )
        st.dataframe(pd.DataFrame(city_rows), use_container_width=True)
else:
    err_msg = arima_metrics_error or "ARIMA baseline metrics not available yet."
    st.caption(err_msg)
