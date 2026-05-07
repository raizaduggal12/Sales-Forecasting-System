"""
Facebook Prophet Model
Handles trend, seasonality (weekly/yearly), and US holidays automatically.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet
import logging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)


class ProphetModel:
    """Thin wrapper around Facebook Prophet for weekly sales forecasting."""

    def __init__(self):
        self.model = None
        self.last_date = None

    def _build_model(self) -> Prophet:
        return Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,   # data is already weekly aggregated
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=10.0,
            interval_width=0.95,
        )

    def fit(self, train_df: pd.DataFrame):
        """
        train_df must have columns: ['Date', 'Sales']
        """
        prophet_df = train_df[['Date', 'Sales']].rename(
            columns={'Date': 'ds', 'Sales': 'y'}
        )
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        prophet_df['y'] = prophet_df['y'].clip(lower=0)

        self.model = self._build_model()
        self.model.add_country_holidays(country_name='US')
        self.model.fit(prophet_df)
        self.last_date = prophet_df['ds'].max()
        return self

    def predict(self, steps: int) -> np.ndarray:
        future = self.model.make_future_dataframe(periods=steps, freq='W')
        forecast = self.model.predict(future)
        # Return only the future predictions (last `steps` rows)
        preds = forecast.tail(steps)['yhat'].values
        return np.maximum(preds, 0)

    def predict_with_dates(self, steps: int) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=steps, freq='W')
        forecast = self.model.predict(future)
        result = forecast.tail(steps)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        result.columns = ['Date', 'Forecast', 'Lower', 'Upper']
        result['Forecast'] = result['Forecast'].clip(lower=0)
        return result.reset_index(drop=True)

    def evaluate(self, val_df: pd.DataFrame) -> dict:
        pred = self.predict(len(val_df))
        actual = val_df['Sales'].values
        mae  = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}