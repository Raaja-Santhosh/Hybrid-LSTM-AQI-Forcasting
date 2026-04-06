"""
Health Risk Classification Training Script
----------------------------------------
Trains Random Forest and Logistic Regression models for AQI health risk classes.
Saves the best model and a metrics report to models/.

Outputs:
- models/health_classifier.joblib
- models/health_classifier_metrics.json
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def aqi_to_risk_bucket(aqi_value: float) -> str:
    if aqi_value <= 50:
        return "Good"
    if aqi_value <= 100:
        return "Moderate"
    if aqi_value <= 200:
        return "Unhealthy"
    return "Hazardous"


def load_training_data() -> pd.DataFrame:
    candidate_files = ["data/master_dataset.csv", "data/city_day.csv"]
    for path in candidate_files:
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            print(f"Loaded dataset: {path} ({len(df):,} rows)")
            return df
    raise FileNotFoundError("No dataset found. Expected data/master_dataset.csv or data/city_day.csv")


def prepare_features(df: pd.DataFrame):
    required_pollutants = ["PM2.5", "PM10", "NO2", "SO2"]
    optional_features = ["CO", "O3", "City"]

    available_features = [c for c in required_pollutants + optional_features if c in df.columns]
    missing_required = [c for c in required_pollutants if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required pollutant columns: {missing_required}")

    if "AQI_Bucket" in df.columns:
        y = df["AQI_Bucket"].astype(str)
    elif "AQI" in df.columns:
        y = df["AQI"].astype(float).apply(aqi_to_risk_bucket)
    else:
        raise ValueError("Need either AQI_Bucket or AQI column for classification target")

    X = df[available_features].copy()

    # Keep standard 4-class objective by mapping variants into nearest category.
    map_rules = {
        "Good": "Good",
        "Satisfactory": "Moderate",
        "Moderate": "Moderate",
        "Poor": "Unhealthy",
        "Very Poor": "Hazardous",
        "Severe": "Hazardous",
        "Unhealthy": "Unhealthy",
        "Hazardous": "Hazardous",
    }
    y = y.map(lambda v: map_rules.get(v, v))

    # Drop rows with unknown labels.
    keep_labels = {"Good", "Moderate", "Unhealthy", "Hazardous"}
    valid_mask = y.isin(keep_labels)
    X = X[valid_mask]
    y = y[valid_mask]

    return X, y, available_features


def reduce_dataset_for_memory(x: pd.DataFrame, y: pd.Series, max_rows: int = 120000):
    if len(x) <= max_rows:
        return x, y

    sampled_idx = y.groupby(y, group_keys=False).apply(
        lambda grp: grp.sample(frac=max_rows / len(y), random_state=42)
    ).index

    x_reduced = x.loc[sampled_idx]
    y_reduced = y.loc[sampled_idx]
    print(f"Downsampled dataset for memory: {len(x):,} -> {len(x_reduced):,} rows")
    return x_reduced, y_reduced


def build_pipelines(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    rf_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=None,
                    random_state=42,
                    n_jobs=1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    lr_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    return {"random_forest": rf_model, "logistic_regression": lr_model}


def evaluate_model(name, model, x_test, y_test):
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"{name}: accuracy={acc:.4f}, f1_weighted={f1:.4f}")
    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "classification_report": report,
    }


def main():
    print("=" * 60)
    print("  AtmosIQ Health Classification Training")
    print("=" * 60)

    df = load_training_data()
    x, y, feature_cols = prepare_features(df)
    x, y = reduce_dataset_for_memory(x, y, max_rows=120000)

    if len(x) < 1000:
        raise ValueError(f"Insufficient training rows after cleaning: {len(x)}")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_cols = [c for c in feature_cols if c != "City"]
    categorical_cols = [c for c in feature_cols if c == "City"]
    models = build_pipelines(numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    metrics = {}
    best_name = None
    best_model = None
    best_score = -np.inf

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(x_train, y_train)
        result = evaluate_model(model_name, model, x_test, y_test)
        metrics[model_name] = result

        if result["f1_weighted"] > best_score:
            best_score = result["f1_weighted"]
            best_name = model_name
            best_model = model

    os.makedirs("models", exist_ok=True)

    artifact = {
        "model_name": best_name,
        "model": best_model,
        "features": feature_cols,
        "labels": ["Good", "Moderate", "Unhealthy", "Hazardous"],
        "metrics": metrics,
    }

    joblib.dump(artifact, "models/health_classifier.joblib")

    metrics_payload = {
        "selected_model": best_name,
        "selected_model_f1_weighted": best_score,
        "all_metrics": metrics,
        "feature_columns": feature_cols,
    }

    with open("models/health_classifier_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\nSaved model artifact: models/health_classifier.joblib")
    print("Saved metrics report: models/health_classifier_metrics.json")


if __name__ == "__main__":
    main()
