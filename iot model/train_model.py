"""
AtmosIQ LSTM Training Script v2.0
──────────────────────────────────
Trains on ALL cities in the master dataset using grouped sequences.
Outputs: models/lstm_model.keras + models/scaler.joblib

INSTRUCTIONS FOR YOUR FRIEND:
1. Put this file and data/master_dataset.csv in the same folder structure.
2. Run: pip install pandas numpy scikit-learn tensorflow joblib
3. Run: python train_model.py
4. Send back the ENTIRE 'models/' folder to the project owner.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("=" * 60)
print("  🌫  AtmosIQ LSTM Training Script v2.0")
print("  📡  Hybrid Context-Aware AQI Forecasting System")
print("=" * 60)

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
file_path = 'data/master_dataset.csv'
if not os.path.exists(file_path):
    print(f"❌ Error: Could not find {file_path}")
    print("   Make sure the data/ folder with master_dataset.csv is next to this script.")
    exit()

print("\n📊 Step 1/5: Loading master dataset...")
df = pd.read_csv(file_path, low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date', 'AQI', 'City'], inplace=True)
df.sort_values(by=['City', 'Date'], inplace=True)

print(f"   Total records loaded: {len(df):,}")
print(f"   Cities found: {df['City'].nunique()}")
print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# ──────────────────────────────────────────────
# 2. BUILD SEQUENCES (Per City, Grouped)
# ──────────────────────────────────────────────
print("\n🔗 Step 2/5: Building 30-day sequences per city...")
SEQUENCE_LENGTH = 30

# Scale the ENTIRE AQI column globally so all cities share the same scale
scaler = MinMaxScaler(feature_range=(0, 1))
df['AQI_scaled'] = scaler.fit_transform(df[['AQI']])

X_all, y_all = [], []

for city, group in df.groupby('City'):
    scaled_values = group['AQI_scaled'].values
    if len(scaled_values) < SEQUENCE_LENGTH + 1:
        continue  # Skip cities with insufficient data
    for i in range(SEQUENCE_LENGTH, len(scaled_values)):
        X_all.append(scaled_values[i - SEQUENCE_LENGTH:i])
        y_all.append(scaled_values[i])

X_all = np.array(X_all)
y_all = np.array(y_all)
X_all = np.reshape(X_all, (X_all.shape[0], X_all.shape[1], 1))

print(f"   Total training sequences created: {len(X_all):,}")

# 80/20 Split
split = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]
print(f"   Training set: {len(X_train):,} | Test set: {len(X_test):,}")

# ──────────────────────────────────────────────
# 3. BUILD THE NEURAL NETWORK
# ──────────────────────────────────────────────
print("\n🧠 Step 3/5: Constructing LSTM Neural Network...")
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# ──────────────────────────────────────────────
# 4. TRAIN
# ──────────────────────────────────────────────
print("\n🚀 Step 4/5: Training starting...")
print("   This will take 5-15 minutes depending on GPU/CPU.")
print("   EarlyStopping enabled — it will auto-stop when the AI peaks.\n")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    batch_size=64,
    epochs=50,
    callbacks=[early_stop],
    verbose=1
)

# ──────────────────────────────────────────────
# 5. EVALUATE & SAVE
# ──────────────────────────────────────────────
print("\n📈 Step 5/5: Evaluating model accuracy...")

# Predict on the test set
predictions_scaled = model.predict(X_test)
predictions = scaler.inverse_transform(predictions_scaled)
actuals = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))

print(f"\n   ╔══════════════════════════════════════╗")
print(f"   ║  MODEL ACCURACY REPORT               ║")
print(f"   ╠══════════════════════════════════════╣")
print(f"   ║  MAE  (Mean Absolute Error): {mae:>7.2f} ║")
print(f"   ║  RMSE (Root Mean Sq Error) : {rmse:>7.2f} ║")
print(f"   ╚══════════════════════════════════════╝")

# Save model + scaler
os.makedirs('models', exist_ok=True)
model.save('models/lstm_model.keras')
joblib.dump(scaler, 'models/scaler.joblib')

print(f"\n✅ SUCCESS! Files saved:")
print(f"   📦 models/lstm_model.keras  (The trained AI brain)")
print(f"   📦 models/scaler.joblib     (The number scaler — REQUIRED by main.py)")
print(f"\n🔁 Transfer the ENTIRE 'models/' folder back to your laptop.")
print(f"   Drop it into the project root and restart FastAPI. Done!")
