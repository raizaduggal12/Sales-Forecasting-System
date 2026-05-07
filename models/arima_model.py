"""
ARIMA / SARIMA Model
Uses auto-selection of (p,d,q)(P,D,Q,s) orders via AIC grid search.
"""

import pandas as pd
import numpy as np
import warnings
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings('ignore')


def _check_stationarity(series: pd.Series) -> int:
    """Return required differencing order d based on ADF test."""
    result = adfuller(series.dropna())
    if result[1] <= 0.05:
        return 0
    result1 = adfuller(series.dropna().diff().dropna())
    if result1[1] <= 0.05:
        return 1
    return 2


def _aic_grid_search(train: pd.Series, seasonal_period: int = 52,
                     max_p: int = 2, max_q: int = 2):
    """
    Lightweight AIC grid search over SARIMA(p,d,q)(P,D,Q,s).
    Returns best (order, seasonal_order).
    """
    d  = _check_stationarity(train)
    D  = 1 if seasonal_period > 1 else 0
    best_aic = np.inf
    best_order = (1, d, 1)
    best_seasonal = (1, D, 0, seasonal_period)

    for p, q in product(range(max_p + 1), range(max_q + 1)):
        for P, Q in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            try:
                mod = SARIMAX(
                    train,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = mod.fit(disp=False)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, d, q)
                    best_seasonal = (P, D, Q, seasonal_period)
            except Exception:
                continue

    return best_order, best_seasonal


class SARIMAModel:
    """Wrapper around statsmodels SARIMAX with auto-order selection."""

    def __init__(self, seasonal_period: int = 52):
        self.seasonal_period = seasonal_period
        self.model_fit = None
        self.order = None
        self.seasonal_order = None

    def fit(self, train_series: pd.Series, auto_order: bool = True):
        if auto_order:
            self.order, self.seasonal_order = _aic_grid_search(
                train_series, self.seasonal_period
            )
        else:
            d = _check_stationarity(train_series)
            self.order = (1, d, 1)
            self.seasonal_order = (1, 1, 0, self.seasonal_period)

        self.model_fit = SARIMAX(
            train_series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, steps: int) -> np.ndarray:
        forecast = self.model_fit.get_forecast(steps=steps)
        return np.maximum(forecast.predicted_mean.values, 0)

    def evaluate(self, val_series: pd.Series) -> dict:
        pred = self.predict(len(val_series))
        actual = val_series.values
        mae  = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}