"""
Data Preprocessing Module
Handles loading, cleaning, and feature engineering for time series forecasting.
"""

import pandas as pd
import numpy as np
import holidays
import warnings
warnings.filterwarnings('ignore')


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load raw Excel data, parse dates, fill missing date gaps per state,
    and return a clean DataFrame.
    """
    df = pd.read_excel(filepath)

    # Parse dates (mixed formats - try dayfirst then fallback)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    # Drop rows where date could not be parsed
    bad_dates = df['Date'].isnull().sum()
    if bad_dates:
        print(f"[WARN] Dropped {bad_dates} rows with unparseable dates.")
    df = df.dropna(subset=['Date'])

    # Sort
    df = df.sort_values(['State', 'Date']).reset_index(drop=True)

    # Rename for clarity
    df = df.rename(columns={'Total': 'Sales'})

    # Fill missing date gaps per state (resample to weekly freq)
    states = df['State'].unique()
    filled_frames = []

    for state in states:
        sub = df[df['State'] == state].set_index('Date')[['Sales']].copy()
        # Resample to weekly; interpolate gaps
        sub = sub.resample('W').mean()
        # FIX: use .bfill().ffill() instead of fillna(method=...)
        sub['Sales'] = sub['Sales'].interpolate(method='linear').bfill().ffill()
        sub['State'] = state
        sub = sub.reset_index()
        filled_frames.append(sub)

    df_clean = pd.concat(filled_frames, ignore_index=True)
    df_clean = df_clean.sort_values(['State', 'Date']).reset_index(drop=True)

    print(f"[INFO] Clean data shape: {df_clean.shape}")
    print(f"[INFO] Date range: {df_clean['Date'].min().date()} to {df_clean['Date'].max().date()}")
    print(f"[INFO] States: {df_clean['State'].nunique()}")
    return df_clean


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer time-based and lag features for ML models (XGBoost, LSTM).
    Also adds a 'holiday_flag' using the US holiday calendar.
    """
    df = df.copy().sort_values(['State', 'Date']).reset_index(drop=True)

    us_holidays = holidays.UnitedStates(years=list(range(2018, 2025)))

    all_frames = []
    for state, grp in df.groupby('State'):
        grp = grp.sort_values('Date').reset_index(drop=True)

        # Calendar features
        grp['day_of_week']  = grp['Date'].dt.dayofweek
        grp['week_of_year'] = grp['Date'].dt.isocalendar().week.astype(int)
        grp['month']        = grp['Date'].dt.month
        grp['quarter']      = grp['Date'].dt.quarter
        grp['year']         = grp['Date'].dt.year
        grp['holiday_flag'] = grp['Date'].apply(lambda d: 1 if d.date() in us_holidays else 0)

        # Lag features
        for lag in [1, 2, 4, 8, 13, 26, 52]:
            grp[f'lag_{lag}'] = grp['Sales'].shift(lag)

        # Rolling statistics
        for window in [4, 8, 13, 26]:
            grp[f'rolling_mean_{window}'] = grp['Sales'].shift(1).rolling(window).mean()
            grp[f'rolling_std_{window}']  = grp['Sales'].shift(1).rolling(window).std()

        # Trend
        grp['trend'] = np.arange(len(grp))

        all_frames.append(grp)

    result = pd.concat(all_frames, ignore_index=True)
    return result


def train_val_split(df_state: pd.DataFrame, val_weeks: int = 8):
    """
    Strict temporal split - no data leakage.
    Returns (train_df, val_df).
    """
    df_state = df_state.sort_values('Date').reset_index(drop=True)
    split_idx = len(df_state) - val_weeks
    return df_state.iloc[:split_idx].copy(), df_state.iloc[split_idx:].copy()


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return ML feature columns (exclude target + id columns)."""
    exclude = {'Date', 'State', 'Sales', 'Category'}
    return [c for c in df.columns if c not in exclude]


if __name__ == '__main__':
    df = load_and_clean_data('data/sales_data.xlsx')
    df_feat = add_features(df)
    print(df_feat.columns.tolist())
    print(df_feat.head())