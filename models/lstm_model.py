"""
LSTM Deep Learning Model
Sequence-to-one LSTM with multi-step recursive forecasting.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler


class LSTMModel:
    """
    LSTM-based time series forecaster.
    Uses a sliding window of `lookback` weeks to predict the next value,
    then recurses for multi-step ahead forecasting.
    """

    def __init__(self, lookback: int = 26, epochs: int = 100, batch_size: int = 16):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_series = None

    def _create_sequences(self, data: np.ndarray):
        """Convert 1D array to (X, y) supervised sequences."""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback: i, 0])
            y.append(data[i, 0])
        return np.array(X).reshape(-1, self.lookback, 1), np.array(y)

    def _build_model(self) -> Sequential:
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae'],
        )
        return model

    def fit(self, train_series: pd.Series):
        self.train_series = train_series.copy()
        values = train_series.values.reshape(-1, 1).astype(float)
        scaled = self.scaler.fit_transform(values)

        if len(scaled) <= self.lookback:
            # If too short, reduce lookback
            self.lookback = max(4, len(scaled) // 2)

        X, y = self._create_sequences(scaled)

        if len(X) == 0:
            raise ValueError("Not enough data to create sequences.")

        # 90/10 internal split for early stopping validation
        split = max(1, int(len(X) * 0.9))
        X_tr, X_vl = X[:split], X[split:]
        y_tr, y_vl = y[:split], y[split:]

        self.model = self._build_model()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=0),
        ]

        validation_data = (X_vl, y_vl) if len(X_vl) > 0 else None

        self.model.fit(
            X_tr, y_tr,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def predict(self, steps: int) -> np.ndarray:
        """Recursive multi-step forecast using the trained LSTM."""
        history = self.train_series.values.astype(float).reshape(-1, 1)
        history_scaled = self.scaler.transform(history).flatten().tolist()

        predictions_scaled = []
        for _ in range(steps):
            window = np.array(history_scaled[-self.lookback:]).reshape(1, self.lookback, 1)
            pred_scaled = float(self.model.predict(window, verbose=0)[0][0])
            predictions_scaled.append(pred_scaled)
            history_scaled.append(pred_scaled)

        predictions = self.scaler.inverse_transform(
            np.array(predictions_scaled).reshape(-1, 1)
        ).flatten()
        return np.maximum(predictions, 0)

    def evaluate(self, val_series: pd.Series) -> dict:
        pred = self.predict(len(val_series))
        actual = val_series.values
        mae  = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}