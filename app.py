import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="AQI Prediction System", layout="wide")

st.title("🌫 Hybrid AQI Forecasting and Health Risk Classification System")
st.markdown("Predict the air quality and health risk index based on historical pollution data.")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    file_path = "data/city_day.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Basic preprocessing for the prototype
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

df = load_data()

# --- SIDEBAR ---
st.sidebar.header("🔧 Settings")
if df is not None:
    cities = df['City'].unique()
    selected_city = st.sidebar.selectbox("Select City", cities)
    
    city_data = df[df['City'] == selected_city].dropna(subset=['AQI'])
    
    st.sidebar.markdown("---")
    st.sidebar.success("Data Loaded Successfully!")
else:
    st.sidebar.error("Error: `data/city_day.csv` not found!")

# --- MAIN DASHBOARD ---
tab1, tab2, tab3 = st.tabs(["📊 Current Data", "🔮 Prediction (AI)", "📈 Historical Trends"])

with tab1:
    st.header(f"Current AQI Overview: {selected_city if df is not None else 'N/A'}")
    if df is not None and not city_data.empty:
        latest_record = city_data.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AQI", latest_record['AQI'])
        col2.metric("PM2.5", latest_record['PM2.5'])
        col3.metric("NO2", latest_record['NO2'])
        col4.metric("Risk Level", latest_record.get('AQI_Bucket', 'Unknown'))
        
        st.subheader("Recent Data (Last 5 Days)")
        st.dataframe(city_data.tail(5)[['Date', 'PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI', 'AQI_Bucket']])
    else:
        st.info("No data available to display.")

with tab2:
    st.header("Future AQI Prediction")
    st.markdown("*(To be implemented by the ML Team)*")
    
    st.info("The LSTM Model and classification prediction will go here.")
    # Placeholder for model prediction
    st.metric("Predicted AQI (Tomorrow)", "TBD", delta="TBD")
    st.warning("Model `lstm_model.h5` not yet loaded.")

with tab3:
    st.header("Historical Pollution Trends")
    if df is not None and not city_data.empty:
        fig = px.line(city_data, x="Date", y="AQI", title=f"Historical AQI for {selected_city}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data to display.")
