"""
FastAPI REST API for Sales Forecasting
Exposes predictions via clean REST endpoints.

Run with:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /                          → health check
    GET  /states                    → list available states
    GET  /forecast/{state}          → next 8-week forecast for a state
    GET  /forecast/{state}?weeks=N  → next N-week forecast
    GET  /metrics/{state}           → model comparison metrics for a state
    GET  /best-model/{state}        → best model name and reason
    POST /retrain                   → trigger retraining (async-compatible)
    GET  /summary                   → summary table of all states
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(BASE_DIR, 'outputs', 'all_results.json')
MODELS_DIR   = os.path.join(BASE_DIR, 'outputs', 'models')
DATA_PATH    = os.path.join(BASE_DIR, 'data', 'sales_data.xlsx')

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sales Forecasting API",
    description="End-to-end time series forecasting: SARIMA, Prophet, XGBoost, LSTM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory cache ───────────────────────────────────────────────────────────
_results_cache: Dict[str, Any] = {}
_models_cache:  Dict[str, Any] = {}
_training_status: str = "idle"   # idle | running | done | error


def load_results() -> Dict[str, Any]:
    global _results_cache
    if _results_cache:
        return _results_cache
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            "No training results found. Run `python train.py` first."
        )
    with open(RESULTS_PATH) as f:
        _results_cache = json.load(f)
    return _results_cache


def load_model(state: str):
    if state in _models_cache:
        return _models_cache[state]
    pkl_path = os.path.join(MODELS_DIR, f"{state.replace(' ', '_')}.pkl")
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)
    _models_cache[state] = model
    return model


def _normalise_state(state: str, results: dict) -> str:
    """Case-insensitive state lookup."""
    for s in results:
        if s.lower() == state.lower():
            return s
    raise HTTPException(status_code=404, detail=f"State '{state}' not found.")


# ── Pydantic response schemas ─────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    states_available: int

class ForecastPoint(BaseModel):
    date: str
    forecast_sales: float

class ForecastResponse(BaseModel):
    state: str
    best_model: str
    forecast_weeks: int
    forecast: List[ForecastPoint]

class MetricsResponse(BaseModel):
    state: str
    best_model: str
    model_metrics: Dict[str, Dict[str, float]]

class SummaryRow(BaseModel):
    state: str
    best_model: str
    rmse: float
    mape: float

class RetrainResponse(BaseModel):
    message: str
    status: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """API health check."""
    try:
        results = load_results()
        n_states = len(results)
    except Exception:
        n_states = 0
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat() + "Z",
        states_available=n_states,
    )


@app.get("/states", tags=["Meta"])
def list_states():
    """List all states with trained models."""
    results = load_results()
    states = [s for s in sorted(results.keys()) if 'error' not in results[s]]
    return {"total": len(states), "states": states}


@app.get("/forecast/{state}", response_model=ForecastResponse, tags=["Forecast"])
def get_forecast(
    state: str,
    weeks: int = Query(default=8, ge=1, le=52, description="Weeks to forecast")
):
    """
    Return sales forecast for the given state.
    - **state**: US state name (e.g. 'California')
    - **weeks**: number of future weeks (default 8, max 52)
    """
    results = load_results()
    state = _normalise_state(state, results)
    record = results[state]

    if 'error' in record:
        raise HTTPException(status_code=500, detail=f"Training error: {record['error']}")

    stored_dates  = record['forecast']['dates']
    stored_values = record['forecast']['values']

    if weeks <= len(stored_dates):
        dates  = stored_dates[:weeks]
        values = stored_values[:weeks]
    else:
        # Extend using the loaded model
        model = load_model(state)
        if model is None:
            raise HTTPException(status_code=404, detail="Model file not found; retrain first.")
        extra = weeks - len(stored_dates)
        extra_preds = model.predict(weeks)
        last_stored = pd.to_datetime(stored_dates[-1])
        extra_dates = [
            (last_stored + pd.Timedelta(weeks=i + 1)).strftime('%Y-%m-%d')
            for i in range(extra)
        ]
        dates  = stored_dates + extra_dates
        values = stored_values + [round(float(v), 2) for v in extra_preds[len(stored_dates):]]

    forecast_points = [
        ForecastPoint(date=d, forecast_sales=v)
        for d, v in zip(dates, values)
    ]

    return ForecastResponse(
        state=state,
        best_model=record['best_model'],
        forecast_weeks=weeks,
        forecast=forecast_points,
    )


@app.get("/metrics/{state}", response_model=MetricsResponse, tags=["Evaluation"])
def get_metrics(state: str):
    """
    Return model comparison metrics (MAE, RMSE, MAPE) for a state.
    """
    results = load_results()
    state = _normalise_state(state, results)
    record = results[state]

    if 'error' in record:
        raise HTTPException(status_code=500, detail=record['error'])

    # Round metrics
    clean_metrics = {}
    for model, metrics in record['models'].items():
        if all(v < 1e11 for v in metrics.values()):
            clean_metrics[model] = {k: round(v, 2) for k, v in metrics.items()}

        return MetricsResponse(
            state=state,
            best_model=record['best_model'],
            model_metrics=clean_metrics,
        )


@app.get("/best-model/{state}", tags=["Evaluation"])
def get_best_model(state: str):
    """Return which model was selected as best and why."""
    results = load_results()
    state = _normalise_state(state, results)
    record = results[state]
    if 'error' in record:
        raise HTTPException(status_code=500, detail=record['error'])

    valid = {
        m: v for m, v in record['models'].items()
        if v.get('RMSE', 1e12) < 1e11
    }
    ranked = sorted(valid.items(), key=lambda x: x[1]['RMSE'])

    return {
        "state": state,
        "best_model": record['best_model'],
        "selection_criterion": "Lowest RMSE on hold-out validation set (last 8 weeks)",
        "ranking": [
            {"model": m, "RMSE": round(v['RMSE'], 2), "MAPE": round(v['MAPE'], 2)}
            for m, v in ranked
        ],
    }


@app.get("/summary", tags=["Meta"])
def get_summary():
    """Summary table: best model and key metrics for every state."""
    results = load_results()
    rows = []
    for state, record in sorted(results.items()):
        if 'error' in record:
            continue
        best = record.get('best_model', 'N/A')
        m = record['models'].get(best, {})
        rows.append({
            "state":       state,
            "best_model":  best,
            "RMSE":        round(m.get('RMSE', 0), 2),
            "MAPE":        round(m.get('MAPE', 0), 2),
            "MAE":         round(m.get('MAE', 0), 2),
        })
    return {"total_states": len(rows), "results": rows}


@app.post("/retrain", response_model=RetrainResponse, tags=["Admin"])
def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Trigger model retraining in the background.
    Clears caches after completion.
    """
    global _training_status

    def _run_training():
        global _results_cache, _models_cache, _training_status
        _training_status = "running"
        try:
            # Import here to avoid circular issues
            from train import train_all_states
            train_all_states(DATA_PATH)
            _results_cache = {}   # invalidate cache
            _models_cache  = {}
            _training_status = "done"
        except Exception as e:
            _training_status = f"error: {e}"

    if _training_status == "running":
        return RetrainResponse(message="Training already in progress.", status="running")

    background_tasks.add_task(_run_training)
    return RetrainResponse(message="Retraining started in background.", status="running")


@app.get("/training-status", tags=["Admin"])
def training_status():
    """Check current training status."""
    return {"status": _training_status}


@app.get("/forecast-all", tags=["Forecast"])
def forecast_all_states(weeks: int = Query(default=8, ge=1, le=52)):
    """Return forecasts for ALL states in one call."""
    results = load_results()
    all_forecasts = {}
    for state, record in results.items():
        if 'error' in record:
            continue
        stored = record['forecast']
        dates  = stored['dates'][:weeks]
        values = stored['values'][:weeks]
        all_forecasts[state] = {
            "best_model": record['best_model'],
            "forecast": [{"date": d, "forecast_sales": v} for d, v in zip(dates, values)]
        }
    return {"forecast_weeks": weeks, "states": all_forecasts}