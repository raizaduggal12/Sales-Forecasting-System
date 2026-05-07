"""
Model Trainer & Selector
Trains all 4 models per state, compares on validation set, picks best.
Saves results to disk.
"""

import os
import json
import pickle
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_preprocessing import load_and_clean_data, train_val_split
from models.arima_model   import SARIMAModel
from models.prophet_model import ProphetModel
from models.xgboost_model import XGBoostModel
from models.lstm_model    import LSTMModel

FORECAST_WEEKS = 8
VAL_WEEKS      = 8
DATA_PATH      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sales_data.xlsx')
SAVE_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(SAVE_DIR, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
def train_state(state: str, df_state: pd.DataFrame,
                val_weeks: int = VAL_WEEKS) -> dict:
    """
    Train all 4 models for one state.
    Returns a results dict with metrics, best model name, and forecast.
    """
    train_df, val_df = train_val_split(df_state, val_weeks=val_weeks)

    train_series = train_df.set_index('Date')['Sales']
    val_series   = val_df.set_index('Date')['Sales']

    results = {'state': state, 'models': {}, 'best_model': None, 'forecast': None}
    trained_models = {}

    # ── 1. SARIMA ────────────────────────────────────────────────────────────
    print(f"  [SARIMA] fitting...", end='', flush=True)
    try:
        sarima = SARIMAModel(seasonal_period=52)
        sarima.fit(train_series, auto_order=False)
        metrics = sarima.evaluate(val_series)
        results['models']['SARIMA'] = metrics
        trained_models['SARIMA'] = sarima
        print(f" RMSE={metrics['RMSE']:,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        results['models']['SARIMA'] = {'MAE': 1e12, 'RMSE': 1e12, 'MAPE': 1e12}

    # ── 2. Prophet ───────────────────────────────────────────────────────────
    print(f"  [Prophet] fitting...", end='', flush=True)
    try:
        prophet = ProphetModel()
        prophet.fit(train_df)
        metrics = prophet.evaluate(val_df)
        results['models']['Prophet'] = metrics
        trained_models['Prophet'] = prophet
        print(f" RMSE={metrics['RMSE']:,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        results['models']['Prophet'] = {'MAE': 1e12, 'RMSE': 1e12, 'MAPE': 1e12}

    # ── 3. XGBoost ───────────────────────────────────────────────────────────
    print(f"  [XGBoost] fitting...", end='', flush=True)
    try:
        xgb_model = XGBoostModel()
        xgb_model.fit(train_series)
        metrics = xgb_model.evaluate(val_series)
        results['models']['XGBoost'] = metrics
        trained_models['XGBoost'] = xgb_model
        print(f" RMSE={metrics['RMSE']:,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        results['models']['XGBoost'] = {'MAE': 1e12, 'RMSE': 1e12, 'MAPE': 1e12}

    # ── 4. LSTM ──────────────────────────────────────────────────────────────
    print(f"  [LSTM] fitting...", end='', flush=True)
    try:
        lstm = LSTMModel(lookback=26, epochs=80)
        lstm.fit(train_series)
        metrics = lstm.evaluate(val_series)
        results['models']['LSTM'] = metrics
        trained_models['LSTM'] = lstm
        print(f" RMSE={metrics['RMSE']:,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        results['models']['LSTM'] = {'MAE': 1e12, 'RMSE': 1e12, 'MAPE': 1e12}

    # ── Select best by RMSE ──────────────────────────────────────────────────
    best_name = min(results['models'], key=lambda k: results['models'][k]['RMSE'])
    results['best_model'] = best_name
    print(f"  Best: {best_name}")

    # ── Re-train best model on FULL data, generate 8-week forecast ───────────
    full_series = df_state.set_index('Date')['Sales']

    if best_name == 'SARIMA':
        best = SARIMAModel(seasonal_period=52)
        best.fit(full_series, auto_order=False)
        preds = best.predict(FORECAST_WEEKS)

    elif best_name == 'Prophet':
        best = ProphetModel()
        best.fit(df_state)
        preds = best.predict(FORECAST_WEEKS)

    elif best_name == 'XGBoost':
        best = XGBoostModel()
        best.fit(full_series)
        preds = best.predict(FORECAST_WEEKS)

    elif best_name == 'LSTM':
        best = LSTMModel(lookback=26, epochs=80)
        best.fit(full_series)
        preds = best.predict(FORECAST_WEEKS)

    else:
        preds = np.full(FORECAST_WEEKS, full_series.mean())
        best = None

    # Build forecast dates
    last_date = df_state['Date'].max()
    forecast_dates = [last_date + pd.Timedelta(weeks=i + 1) for i in range(FORECAST_WEEKS)]
    results['forecast'] = {
        'dates':  [d.strftime('%Y-%m-%d') for d in forecast_dates],
        'values': [round(float(v), 2) for v in preds],
    }

    return results, trained_models.get(best_name)


# ────────────────────────────────────────────────────────────────────────────
def train_all_states(data_path: str = DATA_PATH) -> dict:
    """
    Main training loop over all states.
    Saves:
      - outputs/all_results.json      (metrics + forecasts for every state)
      - outputs/models/<state>.pkl    (best model object per state)
    """
    df = load_and_clean_data(data_path)
    states = sorted(df['State'].unique())
    all_results = {}
    model_dir = os.path.join(SAVE_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)

    for i, state in enumerate(states, 1):
        print(f"\n[{i}/{len(states)}] Training: {state}")
        df_state = df[df['State'] == state].reset_index(drop=True)

        try:
            state_results, best_model = train_state(state, df_state)
            all_results[state] = state_results
            if best_model is not None:
                pkl_path = os.path.join(model_dir, f"{state.replace(' ', '_')}.pkl")
                with open(pkl_path, 'wb') as f:
                    pickle.dump(best_model, f)
        except Exception as e:
            print(f"  ERROR for {state}: {e}")
            all_results[state] = {'state': state, 'error': str(e)}

    # Save consolidated results JSON
    out_path = os.path.join(SAVE_DIR, 'all_results.json')

    def make_serialisable(obj):
        if isinstance(obj, (np.integer, np.int64)):   return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray):                return obj.tolist()
        return obj

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=make_serialisable)

    print(f"\nTraining complete. Results saved to {out_path}")
    return all_results


if __name__ == '__main__':
    train_all_states()