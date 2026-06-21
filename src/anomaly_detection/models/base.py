"""Classe de base abstraite pour tous les détecteurs d'anomalies."""

from __future__ import annotations

import abc
import joblib
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typing import cast

logger = logging.getLogger(__name__)


class AnomalyDetector(abc.ABC):
    """Interface commune à tous les modèles."""

    name: str = "base"

    @abc.abstractmethod
    def fit(self, X: NDArray[np.float32]) -> "AnomalyDetector":
        """Entraîne le modèle sur des données normales."""

    @abc.abstractmethod
    def score_samples(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Retourne un score d'anomalie ∈ [0, 1] pour chaque sample."""

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Modèle sauvegardé → %s", path)

    @classmethod
    def load(cls, path: Path) -> AnomalyDetector:
        model = joblib.load(path)
        return cast(AnomalyDetector, model)
