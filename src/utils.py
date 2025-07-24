# src/utils.py

"""
utils.py

Modul ini berisi fungsi-fungsi bantu seperti visualisasi dan evaluasi metrik model 
untuk time series forecasting (LSTM & ARIMA), serta EDA (Exploratory Data Analysis).
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted", xlabel="Time", ylabel="Value"):
    """
    Menampilkan grafik nilai aktual vs prediksi.

    Parameters:
        y_true (array-like): Nilai aktual
        y_pred (array-like): Nilai prediksi
        title (str): Judul grafik
        xlabel (str): Label sumbu x
        ylabel (str): Label sumbu y
    """
    plt.figure(figsize=(8, 5))
    plt.plot(y_true, label='Actual', marker='o')
    plt.plot(y_pred, label='Predicted', marker='x')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def regression_metrics(y_true, y_pred):
    """
    Menghitung metrik evaluasi regresi: MAE, RMSE, MAPE.

    Parameters:
        y_true (array-like): Nilai aktual
        y_pred (array-like): Nilai prediksi

    Returns:
        dict: berisi MAE, RMSE, dan MAPE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2)
    }

def plot_error_distribution(errors, title="Error Distribution"):
    """
    Visualisasi distribusi error (residual).

    Parameters:
        errors (array-like): Selisih y_true - y_pred
        title (str): Judul grafik
    """
    plt.figure(figsize=(6, 4))
    sns.histplot(errors, kde=True, color="coral")
    plt.title(title)
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
