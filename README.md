# Hybrid LSTM-Based AQI Forecasting

A rapid prototype for predicting the Air Quality Index (AQI) and health risk classification.

## Getting Started

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard:**
   ```bash
   streamlit run app.py
   ```

3. **Run the Backend API (optional but recommended):**
   ```bash
   uvicorn main:app --reload
   ```



5. **Train ARIMA Baseline (optional):**
   ```bash
   python train_arima_baseline.py
   ```

## Workflow
- **Data**: Put your cleaned dataset inside `data/city_day.csv`.
- **Dashboard**: `app.py` is the frontend UI.
- **Model Training**: Create a separate script (e.g. `train.py`) or use Google Colab to train the LSTM and Classification models. Save the models and load them inside `app.py`.

## API Endpoints
- `/api/current-status?city=Delhi`
- `/api/predict?city=Delhi`
- `/api/model-metrics`
- `/api/report-summary?city=Delhi`
- `/api/report-summary/export?city=Delhi`

## Reports
- Exported summary reports: `reports/report_summary_<city>_<timestamp>.json`
