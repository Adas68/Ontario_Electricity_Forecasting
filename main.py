# Install required packages (run this first in Colab)
# !pip install tensorflow statsmodels openpyxl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

print(f"TensorFlow version: {tf.__version__}")

def load_and_preprocess_data(file_path):
    print("Loading and preprocessing data...")
    df_raw = pd.read_excel(file_path, header=None, engine='openpyxl')
    df_clean = df_raw[pd.to_numeric(df_raw.iloc[:, 1], errors='coerce').notnull()].copy()
    df_clean = df_clean.dropna()
    df_clean.columns = ['Date', 'Hour', 'Market_Demand', 'Ontario_Demand']
    df_clean['Hour'] = df_clean['Hour'].astype(int)
    df_clean['Hour_24_fixed'] = df_clean['Hour'].replace(24, 0)
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    df_clean['DateTime'] = pd.to_datetime(df_clean['Date'].astype(str) + ' ' + df_clean['Hour_24_fixed'].astype(str).str.zfill(2) + ':00:00')
    df_clean.loc[df_clean['Hour'] == 24, 'DateTime'] += pd.Timedelta(days=1)
    df_clean.set_index('DateTime', inplace=True)
    data = df_clean[['Ontario_Demand']].astype(float).copy()
    data['HourOfDay'] = data.index.hour
    data['DayOfWeek'] = data.index.dayofweek
    data['Month'] = data.index.month
    data['IsWeekend'] = (data.index.dayofweek >= 5).astype(int)
    print(f"Data range: {data.index.min()} to {data.index.max()}")
    train_data = data.loc[:'2025-06-30 23:00:00']
    test_data = data.loc['2025-07-01 00:00:00':'2025-09-30 23:00:00']
    print(f"Train samples: {len(train_data)} | Test samples: {len(test_data)}")
    return train_data, test_data

def create_sequences(data, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])
    return np.array(X), np.array(y)

def scale_data(train_data, test_data, feature_columns):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data[feature_columns])
    test_scaled = scaler.transform(test_data[feature_columns])
    return scaler, train_scaled, test_scaled

def create_lstm_model(input_shape, neurons=50, dropout=0.2):
    model = Sequential([
        LSTM(neurons, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(dropout),
        LSTM(neurons, activation='relu'),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def create_gru_model(input_shape, neurons=50, dropout=0.2):
    model = Sequential([
        GRU(neurons, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(dropout),
        GRU(neurons, activation='relu'),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def create_tcn_model(input_shape, neurons=50, dropout=0.2):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, padding='causal'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(neurons, activation='relu'),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def create_fnn_model(input_shape, neurons=64, dropout=0.2):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(neurons, activation='relu'),
        Dropout(dropout),
        Dense(neurons, activation='relu'),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def create_bidirectional_lstm_model(input_shape, neurons=50, dropout=0.2):
    model = Sequential([
        Bidirectional(LSTM(neurons, activation='relu', input_shape=input_shape, return_sequences=True)),
        Dropout(dropout),
        Bidirectional(LSTM(neurons, activation='relu')),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, scaler, feature_columns):
    print(f"\n--- Training {model_name} ---")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)
    y_pred = model.predict(X_test)
    pred_reshaped = np.zeros((y_pred.shape[0], len(feature_columns)))
    pred_reshaped[:, 0] = y_pred[:, 0]
    test_reshaped = np.zeros((y_test.shape[0], len(feature_columns)))
    test_reshaped[:, 0] = y_test
    y_pred_inv = scaler.inverse_transform(pred_reshaped)[:, 0]
    y_test_inv = scaler.inverse_transform(test_reshaped)[:, 0]
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label="Actual", color="blue", linewidth=2)
    plt.plot(y_pred_inv, label="Predicted", color="red", linestyle="--", linewidth=2)
    plt.title(f"{model_name}: Actual vs Predicted Demand", fontsize=16)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Demand (MW)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{model_name}_results.png", dpi=150)
    plt.show()
    return mae, rmse, history

def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    file_path = 'PUB_Demand.csv.xlsx'
    train_data, test_data = load_and_preprocess_data(file_path)
    feature_columns = ['Ontario_Demand', 'HourOfDay', 'DayOfWeek', 'Month', 'IsWeekend']
    scaler, train_scaled, test_scaled = scale_data(train_data, test_data, feature_columns)
    seq_length = 24
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    X_train = X_train.reshape((X_train.shape[0], seq_length, len(feature_columns)))
    X_test = X_test.reshape((X_test.shape[0], seq_length, len(feature_columns)))
    models = {
        'LSTM': create_lstm_model,
        'GRU': create_gru_model,
        'TCN': create_tcn_model,
        'FNN': create_fnn_model,
        'Bidirectional LSTM': create_bidirectional_lstm_model
    }
    results = {}
    for name, model_fn in models.items():
        print(f"\n{'='*60}")
        print(f"Creating {name} model...")
        model = model_fn(input_shape=(seq_length, len(feature_columns)))
        mae, rmse, _ = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, name, scaler, feature_columns)
        results[name] = {'MAE': mae, 'RMSE': rmse}
    print("\n" + "="*60)
    print("=== FINAL MODEL COMPARISON ===")
    print("="*60)
    for name, metrics in results.items():
        print(f"{name:20s}: MAE={metrics['MAE']:7.2f}, RMSE={metrics['RMSE']:7.2f}")
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    mae_values = [results[name]['MAE'] for name in model_names]
    rmse_values = [results[name]['RMSE'] for name in model_names]
    x = np.arange(len(model_names))
    width = 0.35
    plt.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8)
    plt.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.8)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.show()
    print("\nCode execution complete. Check generated plots and metrics.")

if __name__ == "__main__":
    main()