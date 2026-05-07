"""
XGBoost Model with Lag + Calendar Features
Uses a supervised regression framing: predict t+1 .. t+8 iteratively.
"""

import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import holidays


# ── Feature engineering helpers ───────────────────────────────────────────────

def _build_features_from_series(series: pd.Series) -> pd.DataFrame:
    """
    Given a sorted weekly Sales series (DatetimeIndex),
    return a feature DataFrame aligned 1-step ahead.
    """
    us_holidays = holidays.UnitedStates(years=list(range(2018, 2026)))
    df = pd.DataFrame({'Sales': series.values}, index=series.index)

    # Calendar
    df['day_of_week']  = df.index.dayofweek
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['month']        = df.index.month
    df['quarter']      = df.index.quarter
    df['year']         = df.index.year
    df['holiday_flag'] = [1 if d.date() in us_holidays else 0 for d in df.index]
    df['trend']        = np.arange(len(df))

    # Lag features
    for lag in [1, 2, 4, 8, 13, 26, 52]:
        df[f'lag_{lag}'] = df['Sales'].shift(lag)

    # Rolling stats (using only past data – shift(1) prevents leakage)
    for w in [4, 8, 13, 26]:
        df[f'roll_mean_{w}'] = df['Sales'].shift(1).rolling(w).mean()
        df[f'roll_std_{w}']  = df['Sales'].shift(1).rolling(w).std()
        df[f'roll_min_{w}']  = df['Sales'].shift(1).rolling(w).min()
        df[f'roll_max_{w}']  = df['Sales'].shift(1).rolling(w).max()

    return df


class XGBoostModel:
    """XGBoost regressor for multi-step ahead weekly forecasting."""

    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.scaler = None
        self.train_series = None   # kept for recursive inference

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        return [c for c in df.columns if c != 'Sales']

    def fit(self, train_series: pd.Series):
        """
        train_series: pd.Series with DatetimeIndex, values = Sales
        """
        self.train_series = train_series.copy()
        df = _build_features_from_series(train_series)
        df = df.dropna()

        self.feature_cols = self._get_feature_cols(df)
        X = df[self.feature_cols].values
        y = df['Sales'].values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.model.fit(X_scaled, y,
                       eval_set=[(X_scaled, y)],
                       verbose=False)
        return self

    def _make_next_row(self, history: pd.Series, future_date: pd.Timestamp) -> np.ndarray:
        """Build one row of features for a future date using rolling history."""
        us_holidays = holidays.UnitedStates(years=[future_date.year])

        feat = {
            'day_of_week':  future_date.dayofweek,
            'week_of_year': future_date.isocalendar()[1],
            'month':        future_date.month,
            'quarter':      (future_date.month - 1) // 3 + 1,
            'year':         future_date.year,
            'holiday_flag': 1 if future_date.date() in us_holidays else 0,
            'trend':        len(history),
        }
        lags = [1, 2, 4, 8, 13, 26, 52]
        for lag in lags:
            feat[f'lag_{lag}'] = history.iloc[-lag] if len(history) >= lag else np.nan
        for w in [4, 8, 13, 26]:
            past = history.iloc[-w:] if len(history) >= w else history
            feat[f'roll_mean_{w}'] = past.mean()
            feat[f'roll_std_{w}']  = past.std()
            feat[f'roll_min_{w}']  = past.min()
            feat[f'roll_max_{w}']  = past.max()

        # Align to stored feature order
        row = np.array([feat.get(c, 0.0) for c in self.feature_cols]).reshape(1, -1)
        return row

    def predict(self, steps: int) -> np.ndarray:
        """Recursive multi-step forecast."""
        history = self.train_series.copy()
        last_date = history.index[-1]
        predictions = []

        for i in range(steps):
            future_date = last_date + pd.Timedelta(weeks=i + 1)
            row = self._make_next_row(history, future_date)
            row_scaled = self.scaler.transform(row)
            pred = float(self.model.predict(row_scaled)[0])
            pred = max(pred, 0)
            predictions.append(pred)
            # Append prediction to history for next step
            new_row = pd.Series([pred], index=[future_date])
            history = pd.concat([history, new_row])

        return np.array(predictions)

    def evaluate(self, val_series: pd.Series) -> dict:
        pred = self.predict(len(val_series))
        actual = val_series.values
        mae  = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}