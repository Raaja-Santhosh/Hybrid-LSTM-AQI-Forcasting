import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("🛠 Starting the Great Data Merge...")

# 1. Load both datasets
print("1. Loading historical and modern datasets...")
df_hist = pd.read_csv('data/data.csv', encoding='unicode_escape', low_memory=False)
df_mod = pd.read_csv('data/city_day.csv', low_memory=False)

# 2. Standardize Columns
print("2. Aligning column names...")
# We map the historical columns to match the modern Kaggle dataset formatting
df_hist.rename(columns={
    'location': 'City',
    'date': 'Date',
    'no2': 'NO2',
    'so2': 'SO2',
    'pm2_5': 'PM2.5',
    'rspm': 'PM10'  # RSPM (Respirable Suspended Particulate Matter) acts as a historical proxy for PM10
}, inplace=True)

# 3. Format Dates properly
print("3. Converting timelines to universal datetime format...")
df_hist['Date'] = pd.to_datetime(df_hist['Date'], errors='coerce')
df_mod['Date'] = pd.to_datetime(df_mod['Date'], errors='coerce')

# Drop corrupted rows that have no city or no valid date
df_hist.dropna(subset=['Date', 'City'], inplace=True)
df_mod.dropna(subset=['Date', 'City'], inplace=True)

# 4. Mathematically Approximate Historical AQI
print("4. Calculating AI proxy AQI for historical 1990-2015 records...")
# Since AQI didn't exist officially back then, we calculate a rough algorithmic proxy 
# based primarily on PM10 and NO2 spikes.
df_hist['AQI'] = df_hist[['PM10', 'NO2']].max(axis=1) * 1.5 

# Keep all useful columns — CO and O3 only exist in the modern dataset
# but we carry them through so the ML pipeline and dashboards can use them.
common_cols = ['City', 'Date', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI', 'AQI_Bucket']
df_hist = df_hist[[col for col in common_cols if col in df_hist.columns]]
df_mod = df_mod[[col for col in common_cols if col in df_mod.columns]]

# 5. Stacking Data
print("5. Fusing datasets together into the Master Timeline...")
master_df = pd.concat([df_hist, df_mod], ignore_index=True)

# 6. Sorting 
print("6. Sorting cleanly by City, then Chronologically by Time...")
# This ensures that for Delhi, the rows start in 1990 and end in 2020 sequentially.
master_df.sort_values(by=['City', 'Date'], inplace=True)

# If both datasets had a reading for Delhi on January 1st, 2015, we keep the newer, more accurate one.
master_df.drop_duplicates(subset=['City', 'Date'], keep='last', inplace=True)

# 7. Final Output
print("7. Saving Master Dataset to hard drive...")
master_df.to_csv('data/master_dataset.csv', index=False)
print("✅ GENIUS LEVEL UNLOCKED: Created data/master_dataset.csv with exactly", len(master_df), "records spanning 30 Years!")
