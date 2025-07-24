# src/model.py

"""
model.py

Modul ini digunakan untuk melakukan pelatihan dan evaluasi model time series forecasting
(LSTM dan ARIMA) untuk masing-masing tempat wisata.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

def create_sequences(data, n_steps=2):
    """
    Membuat input sequence untuk LSTM.

    Parameters:
        data (array-like): Data terstandardisasi
        n_steps (int): Jumlah langkah input

    Returns:
        X, y: array fitur dan target
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

def train_lstm(data_scaled, n_steps=2):
    """
    Melatih model LSTM berdasarkan data yang sudah diskalakan.

    Parameters:
        data_scaled: data hasil MinMaxScaler
        n_steps: panjang sequence untuk input

    Returns:
        model terlatih, X, y (data untuk evaluasi)
    """
    X, y = create_sequences(data_scaled, n_steps)
    if len(X) == 0:
        return None, None, None

    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)

    return model, X, y

def evaluate_lstm(model, X, y, scaler):
    """
    Mengevaluasi performa model LSTM.

    Returns:
        dict: MAE, RMSE, MAPE
    """
    y_pred_scaled = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'MAE_LSTM': round(mae, 2),
        'RMSE_LSTM': round(rmse, 2),
        'MAPE_LSTM': round(mape, 2)
    }

def predict_next_lstm(model, last_sequence, scaler):
    """
    Memprediksi nilai berikutnya dari sequence terakhir.

    Returns:
        nilai prediksi (real value)
    """
    pred_scaled = model.predict(last_sequence)
    pred = scaler.inverse_transform(pred_scaled)
    return float(pred[0][0])

def train_and_evaluate_arima(series):
    """
    Melatih dan mengevaluasi ARIMA untuk satu tempat wisata.

    Returns:
        prediksi berikutnya dan metrik evaluasi
    """
    try:
        model = ARIMA(series, order=(2, 1, 0))
        fitted = model.fit()
        forecast = fitted.forecast(steps=1)
        next_pred = forecast.iloc[0]

        y_pred = fitted.predict(start=1, end=len(series)-1, typ='levels')
        y_true = series.values[1:]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {
            'next_pred': int(next_pred),
            'MAE_ARIMA': round(mae, 2),
            'RMSE_ARIMA': round(rmse, 2),
            'MAPE_ARIMA': round(mape, 2)
        }

    except:
        return {
            'next_pred': None,
            'MAE_ARIMA': None,
            'RMSE_ARIMA': None,
            'MAPE_ARIMA': None
        }
