"""
IoT Data Stream Simulator
-------------------------
Polls backend APIs periodically to simulate IoT telemetry ingestion and forecast generation.
Writes logs to reports/iot_simulation_log.csv.
"""

import argparse
import os
import time
from datetime import datetime

import pandas as pd
import requests


def fetch_json(method, url, params=None, timeout=15):
    if method.lower() == "post":
        response = requests.post(url, params=params, timeout=timeout)
    else:
        response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def run_simulation(base_url, cities, interval_seconds, iterations):
    rows = []
    print("Starting IoT simulation...")
    print(f"Base URL: {base_url}")
    print(f"Cities: {cities}")
    print(f"Interval: {interval_seconds}s")
    print(f"Iterations: {iterations}")

    for i in range(iterations):
        print(f"\nIteration {i + 1}/{iterations}")
        for city in cities:
            now = datetime.utcnow().isoformat() + "Z"
            current = fetch_json("get", f"{base_url}/api/current-status", params={"city": city})
            pred = fetch_json("post", f"{base_url}/api/predict", params={"city": city})

            p24 = pred.get("predictions", {}).get("24h", {}).get("aqi", pred.get("predicted_aqi_tomorrow"))
            p48 = pred.get("predictions", {}).get("48h", {}).get("aqi")
            p72 = pred.get("predictions", {}).get("72h", {}).get("aqi")

            row = {
                "timestamp": now,
                "city": city,
                "source": current.get("source"),
                "aqi_now": current.get("aqi"),
                "pm25": current.get("pm25"),
                "pm10": current.get("pm10"),
                "no2": current.get("no2"),
                "so2": current.get("so2"),
                "risk_now": (current.get("health_risk") or {}).get("level"),
                "aqi_24h": p24,
                "aqi_48h": p48,
                "aqi_72h": p72,
            }
            rows.append(row)
            print(f"  {city}: AQI now={row['aqi_now']} | +24h={row['aqi_24h']} | risk={row['risk_now']}")

        if i < iterations - 1:
            time.sleep(interval_seconds)

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", "iot_simulation_log.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSimulation complete. Log saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IoT-style AQI polling simulation")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend API base URL")
    parser.add_argument("--cities", default="Delhi,Mumbai,Bangalore", help="Comma-separated city names")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds")
    parser.add_argument("--iterations", type=int, default=10, help="Number of polling rounds")

    args = parser.parse_args()
    city_list = [c.strip() for c in args.cities.split(",") if c.strip()]
    run_simulation(args.base_url.rstrip("/"), city_list, args.interval, args.iterations)
