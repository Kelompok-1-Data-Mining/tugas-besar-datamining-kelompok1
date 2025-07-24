# src/main.py

"""
Main pipeline for Time Series Forecasting (LSTM & ARIMA)
Steps:
1. Load processed data
2. Preprocess (sort dan scaling)
3. Train LSTM & ARIMA per tempat wisata
4. Evaluate and visualize
5. Save prediction result
"""

import pandas as pd
import numpy as np
from data_loader import load_csv
from model import train_lstm, predict_next_lstm, evaluate_lstm, train_and_evaluate_arima
from utils import plot_actual_vs_predicted, regression_metrics, plot_error_distribution
from sklearn.preprocessing import MinMaxScaler

def main():
    # 1. Load data
    try:
        df = load_csv("avg_combined_dataset.csv", processed=True)
    except FileNotFoundError:
        print("Dataset tidak ditemukan.")
        return

    df['year'] = pd.to_datetime(df['year'], format="%Y")
    df = df.sort_values(by=['place_name', 'year'])

    predictions = []

    for place in df['place_name'].unique():
        df_place = df[df['place_name'] == place].copy()

        if len(df_place) < 3:
            print(f"Tempat '{place}' dilewati karena jumlah data < 3.")
            continue

        data_ts = df_place[['total_visitor']].values
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_ts)

        # Train LSTM
        lstm_model, X_lstm, y_lstm = train_lstm(data_scaled)

        if lstm_model is None:
            print(f"Tempat '{place}' dilewati karena data kurang untuk LSTM.")
            continue

        lstm_metrics = evaluate_lstm(lstm_model, X_lstm, y_lstm, scaler)
        next_lstm_pred = predict_next_lstm(lstm_model, data_scaled[-2:].reshape((1, 2, 1)), scaler)

        # Train ARIMA
        arima_result = train_and_evaluate_arima(df_place['total_visitor'])

        year_next = df_place['year'].dt.year.max() + 1
        avg_rating = df_place['avg_rating'].mean()

        # Gabungkan hasil
        predictions.append({
            'place_name': place,
            'predicted_year': year_next,
            'predicted_visitors_lstm': int(next_lstm_pred),
            'predicted_visitors_arima': arima_result['next_pred'],
            'avg_rating': round(avg_rating, 2),
            **lstm_metrics,
            'MAE_ARIMA': arima_result['MAE_ARIMA'],
            'RMSE_ARIMA': arima_result['RMSE_ARIMA'],
            'MAPE_ARIMA': arima_result['MAPE_ARIMA'],
        })

    # Simpan hasil prediksi ke file
    df_result = pd.DataFrame(predictions)
    df_result.to_csv("data/predictions_result.csv", index=False)
    print("Prediksi selesai. Hasil disimpan ke 'data/predictions_result.csv'.")

if __name__ == "__main__":
    main()
