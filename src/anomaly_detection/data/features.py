"""Feature engineering : rolling statistics, lags, normalisation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from numpy.typing import NDArray

from anomaly_detection.config import settings

logger = logging.getLogger(__name__)

def add_rolling_features(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Ajoute des statistiques glissantes (mean, std, min, max) pour chaque fenêtre."""
    cols = cols or [c for c in df.columns if df[c].dtype in (np.float64, np.float32, np.int64)]
    result = df.copy()

    for w in settings.rolling_windows:
        for col in cols:
            rolled = df[col].rolling(window=w, min_periods=1)
            result[f"{col}_roll{w}_mean"] = rolled.mean()
            result[f"{col}_roll{w}_std"] = rolled.std().fillna(0)
            result[f"{col}_roll{w}_max"] = rolled.max()
            result[f"{col}_roll{w}_min"] = rolled.min()

    new_cols = result.shape[1] - df.shape[1]
    logger.info(
        "Rolling features ajoutées : +%d colonnes (fenêtres %s)",
        new_cols,
        settings.rolling_windows,
    )
    return pd.DataFrame(result)


def add_lag_features(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Ajoute des features décalées dans le temps."""
    cols = cols or [c for c in df.columns if df[c].dtype in (np.float64, np.float32, np.int64)]
    result = df.copy()

    for lag in settings.lag_periods:
        for col in cols:
            result[f"{col}_lag{lag}"] = df[col].shift(lag).fillna(0)

    new_cols = result.shape[1] - df.shape[1]
    logger.info("Lag features ajoutées : +%d colonnes (lags %s)", new_cols, settings.lag_periods)
    return pd.DataFrame(result)

def normalize(
    X_train: NDArray[np.float32],
    X_test: NDArray[np.float32] | None = None,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32] | None,
    RobustScaler,
]:
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)

    X_test_scaled: NDArray[np.float32] | None = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test).astype(np.float32)

    logger.info("RobustScaler appliqué — %d features", X_train.shape[1])
    return X_train_scaled, X_test_scaled, scaler


def build_sequences(
    X: NDArray[np.float32],
    seq_len: int,
) -> NDArray[np.float32]:
    n = len(X) - seq_len + 1
    seqs = np.stack([X[i : i + seq_len] for i in range(n)]).astype(np.float32)
    logger.info("Séquences LSTM : %s → %s", X.shape, seqs.shape)
    return seqs