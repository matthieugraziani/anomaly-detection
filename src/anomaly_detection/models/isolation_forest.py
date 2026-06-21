"""Isolation Forest."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from anomaly_detection.config import settings
from anomaly_detection.models.base import AnomalyDetector

logger = logging.getLogger(__name__)


class IsolationForestDetector(AnomalyDetector):
    """Détecteur basé sur Isolation Forest (sklearn)."""

    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = settings.if_n_estimators,
        contamination: float = settings.if_contamination,
        max_features: float = settings.if_max_features,
    ) -> None:
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=max_features,
            random_state=settings.seed,
            n_jobs=-1,
        )
        self._scaler = MinMaxScaler()
        self._fitted = False

    def fit(self, X: NDArray[np.float32]) -> IsolationForestDetector:
        self._model.fit(X)

        raw = np.asarray(
            -self._model.score_samples(X),
            dtype=np.float32,
        )

        self._scaler.fit(raw.reshape(-1, 1))
        self._fitted = True

        logger.info("IsolationForest entraîné sur %d échantillons", len(X))
        return self

    def score_samples(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        if not self._fitted:
            raise RuntimeError("Appelez fit() avant score_samples()")

        raw = np.asarray(
            -self._model.score_samples(X),
            dtype=np.float32,
        )

        self._scaler.fit(raw.reshape(-1, 1))
        self._fitted = True

        logger.info("IsolationForest entraîné sur %d échantillons", len(X))
        return self

    def score_samples(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        if not self._fitted:
            raise RuntimeError("Appelez fit() avant score_samples()")

        raw = np.asarray(
            -self._model.score_samples(X),
            dtype=np.float32,
        )

        scores = self._scaler.transform(raw.reshape(-1, 1)).flatten()

        return np.clip(scores, 0.0, 1.0).astype(np.float32)
