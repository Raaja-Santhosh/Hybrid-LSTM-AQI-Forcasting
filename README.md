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

## Workflow
- **Data**: Put your cleaned dataset inside `data/city_day.csv`.
- **Dashboard**: `app.py` is the frontend UI.
- **Model Training**: Create a separate script (e.g. `train.py`) or use Google Colab to train the LSTM and Classification models. Save the models and load them inside `app.py`.
