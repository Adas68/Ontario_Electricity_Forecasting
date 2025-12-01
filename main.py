"""
Ontario Electricity Demand Forecasting - Final Demo Code
Group 2: Apon Das, Ritwick Vemula, Rayyan Nezami
Course: AISE4010
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import json
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 1. DATA LOADING & CLEANING
# -----------------------------
def load_and_clean_data(filepath):
    print("🔍 Loading and cleaning data...")
    # Load raw Excel file (no header assumed)
    df_raw = pd.read_excel(filepath, header=None)
    
    # Keep only rows where column 1 (Hour) is numeric → this skips metadata like "\\Hourly Demand Report"
    df_raw = df_raw[pd.to_numeric(df_raw.iloc[:, 1], errors='coerce').notnull()]
    df_raw = df_raw.dropna().reset_index(drop=True)
    df_raw.columns = ['Date', 'Hour', 'Market_Demand', 'Ontario_Demand']
    
    # Convert Hour to int
    df_raw['Hour'] = df_raw['Hour'].astype(int)
    
    # Handle Hour=24 → treat as 00:00 next day
    df_raw['Hour_adj'] = df_raw['Hour'].replace(24, 0)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_raw['DateTime'] = pd.to_datetime(
        df_raw['Date'].astype(str) + ' ' + df_raw['Hour_adj'].astype(str).str.zfill(2) + ':00:00'
    )
    # Shift DateTime by 1 day where original Hour was 24
    df_raw.loc[df_raw['Hour'] == 24, 'DateTime'] += pd.Timedelta(days=1)
    
    df_raw.set_index('DateTime', inplace=True)
    # Keep only Ontario_Demand as target
    data = df_raw[['Ontario_Demand']].astype(float).copy()
    
    # Add time features
    data['HourOfDay'] = data.index.hour
    data['DayOfWeek'] = data.index.dayofweek
    data['Month'] = data.index.month
    data['IsWeekend'] = (data.index.dayofweek >= 5).astype(int)
    
    print(f"✅ Data loaded: {data.index.min()} to {data.index.max()}")
    return data

# -----------------------------
# 2. CREATE SEQUENCES
# -----------------------------
def create_sequences(data, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])  # predict Ontario_Demand
    return np.array(X), np.array(y)

# -----------------------------
# 3. MAIN EXECUTION
# -----------------------------
def main():
    FILE = 'PUB_Demand.csv.xlsx'
    
    # Ensure results folder exists
    os.makedirs('results', exist_ok=True)
    
    # Load and clean data
    full_data = load_and_clean_data(FILE)
    
    # Split: Train = Jan 1 – Jun 30, 2025 | Test = Jul 1 – Sep 30, 2025
    train_data = full_data.loc[:'2025-06-30 23:00:00']
    test_data = full_data.loc['2025-07-01 00:00:00':'2025-09-30 23:00:00']
    
    print(f"📊 Train samples: {len(train_data)} | Test samples: {len(test_data)}")
    
    if len(test_data) == 0:
        raise ValueError("❌ Test set is empty! Check your date range.")
    
    # Define features (must match order used in scaling)
    feature_cols = ['Ontario_Demand', 'HourOfDay', 'DayOfWeek', 'Month', 'IsWeekend']
    train_vals = train_data[feature_cols].values
    test_vals = test_data[feature_cols].values
    
    # Scale data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_vals)
    test_scaled = scaler.transform(test_vals)
    
    # Create sequences (past 24 hours → predict next hour)
    SEQ_LEN = 24
    X_train, y_train = create_sequences(train_scaled, SEQ_LEN)
    X_test, y_test = create_sequences(test_scaled, SEQ_LEN)
    
    print(f"🔁 Sequences → Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Build LSTM model
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(SEQ_LEN, len(feature_cols))),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    print("🧠 Training LSTM (20 epochs)...")
    
    # Train
    model.fit(X_train, y_train, epochs=20, batch_size=32, 
              validation_data=(X_test, y_test), verbose=0)
    
    # Predict
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Inverse transform predictions
    def inverse_transform_ontario(scaler, y_array):
        dummy = np.zeros((len(y_array), len(feature_cols)))
        dummy[:, 0] = y_array
        return scaler.inverse_transform(dummy)[:, 0]
    
    y_pred_inv = inverse_transform_ontario(scaler, y_pred)
    y_test_inv = inverse_transform_ontario(scaler, y_test)
    
    # Compute LSTM metrics
    mae_lstm = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    
    # Baseline: Predict demand from 24 hours ago (same hour yesterday)
    baseline_pred = test_data['Ontario_Demand'].shift(24).dropna().values
    baseline_true = test_data['Ontario_Demand'][24:].values
    mae_baseline = mean_absolute_error(baseline_true, baseline_pred)
    rmse_baseline = np.sqrt(mean_squared_error(baseline_true, baseline_pred))
    
    # Print results
    print("\n✅ FINAL RESULTS")
    print(f"📈 LSTM      → MAE: {mae_lstm:.2f} MW | RMSE: {rmse_lstm:.2f} MW")
    print(f"🧱 Baseline  → MAE: {mae_baseline:.2f} MW | RMSE: {rmse_baseline:.2f} MW")
    
    # Save metrics
    metrics = {
        "LSTM_MAE": float(mae_lstm),
        "LSTM_RMSE": float(rmse_lstm),
        "Baseline_MAE": float(mae_baseline),
        "Baseline_RMSE": float(rmse_baseline),
        "Train_Start": str(train_data.index.min()),
        "Train_End": str(train_data.index.max()),
        "Test_Start": str(test_data.index.min()),
        "Test_End": str(test_data.index.max())
    }
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot first 200 predictions
    plt.figure(figsize=(12, 5))
    plt.plot(y_test_inv[:200], label='Actual (MW)', color='steelblue')
    plt.plot(y_pred_inv[:200], label='LSTM Prediction (MW)', color='crimson', linestyle='--')
    plt.title('Ontario Hourly Demand Forecast (First 200 Test Samples)')
    plt.xlabel('Time Step (Hours)')
    plt.ylabel('Demand (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/forecast_results.png', dpi=150)
    plt.close()
    
    print("\n📁 Outputs saved to 'results/' folder:")
    print("   - metrics.json")
    print("   - forecast_results.png")
    print("\n🎉 Code demo ready! All components cohesive and functional.")

if __name__ == "__main__":
    main()