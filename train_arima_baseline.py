"""
ARIMA Baseline Training Script
-----------------------------
Builds ARIMA baselines for top cities and stores evaluation metrics for comparison
against LSTM results.

Outputs:
- models/arima_baseline_metrics.json
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


def evaluate_city_arima(series, order=(2, 1, 2), holdout=72):
    clean = pd.Series(series).dropna().astype(float).values
    if len(clean) < holdout + 40:
        return None

    train = clean[:-holdout]
    test = clean[-holdout:]

    model = ARIMA(train, order=order)
    fitted = model.fit()
    preds = fitted.forecast(steps=holdout)

    mae = float(mean_absolute_error(test, preds))
    rmse = float(np.sqrt(mean_squared_error(test, preds)))

    return {
        "mae": mae,
        "rmse": rmse,
        "holdout": int(holdout),
        "last_actual": float(test[-1]),
        "last_predicted": float(preds[-1]),
    }


def main():
    data_path = "data/master_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Expected data/master_dataset.csv")

    df = pd.read_csv(data_path, low_memory=False)
    df = df.dropna(subset=["City", "AQI"])

    # Focus baseline training on cities with enough history.
    top_cities = (
        df.groupby("City").size().sort_values(ascending=False).head(8).index.tolist()
    )

    per_city = {}
    for city in top_cities:
        city_series = df[df["City"] == city]["AQI"]
        result = evaluate_city_arima(city_series)
        if result is not None:
            per_city[city] = result
            print(f"{city}: MAE={result['mae']:.2f}, RMSE={result['rmse']:.2f}")
        else:
            print(f"{city}: skipped (insufficient history)")

    if not per_city:
        raise RuntimeError("No city had sufficient data to train ARIMA baseline")

    mae_vals = [v["mae"] for v in per_city.values()]
    rmse_vals = [v["rmse"] for v in per_city.values()]

    payload = {
        "model": "ARIMA",
        "order": [2, 1, 2],
        "cities_trained": len(per_city),
        "aggregate": {
            "mae_mean": float(np.mean(mae_vals)),
            "rmse_mean": float(np.mean(rmse_vals)),
            "mae_median": float(np.median(mae_vals)),
            "rmse_median": float(np.median(rmse_vals)),
        },
        "per_city": per_city,
    }

    os.makedirs("models", exist_ok=True)
    out_path = "models/arima_baseline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved ARIMA metrics: {out_path}")


if __name__ == "__main__":
    main()
