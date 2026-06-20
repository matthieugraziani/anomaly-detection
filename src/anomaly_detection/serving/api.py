"""API REST FastAPI pour la détection d'anomalies en temps réel."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from anomaly_detection.config import settings
from anomaly_detection.models.base import AnomalyDetector


logger = logging.getLogger(__name__)

class PredictRequest(BaseModel):
    """Corps de la requête de prédiction."""

    features: list[list[float]] = Field(
        ...,
        description="Matrice de features — shape (n_samples, n_features)",
        examples=[[[0.1, -1.2, 0.8, 0.3]]],
    )
    model_name: str = Field(default="isolation_forest", description="Modèle à utiliser")


class PredictResponse(BaseModel):
    """Réponse de l'API."""

    scores: list[float] = Field(..., description="Score d'anomalie ∈ [0, 1]")
    labels: list[int] = Field(..., description="1 = anomalie, 0 = normal")
    threshold: float = Field(..., description="Seuil appliqué")
    model: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]


# Cache des modèles chargés en mémoire
_model_cache: dict[str, AnomalyDetector] = {}
_thresholds: dict[str, float] = {}


def _load_model(name: str) -> AnomalyDetector:
    """Charge un modèle depuis le disque (lazy loading avec cache)."""
    if name not in _model_cache:
        path = Path("models") / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {path}")
        _model_cache[name] = AnomalyDetector.load(path)
        logger.info("Modèle '%s' chargé en cache", name)
    return _model_cache[name]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Charge les modèles disponibles au démarrage."""
    model_names = ["isolation_forest", "autoencoder", "lstm_ae"]
    for name in model_names:
        try:
            _load_model(name)
            _thresholds[name] = settings.default_threshold
        except FileNotFoundError:
            logger.warning("Modèle '%s' non disponible (lancer le benchmark d'abord)", name)
    logger.info("API prête — modèles chargés : %s", list(_model_cache))
    yield
    _model_cache.clear()


app = FastAPI(
    title="Anomaly Detection API",
    description="Détection d'anomalies temps réel — Isolation Forest, Autoencoder, LSTM-AE",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", models_loaded=list(_model_cache))


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest) -> PredictResponse:
    """Prédit les anomalies pour un batch de samples."""
    try:
        model = _load_model(req.model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    X = np.array(req.features, dtype=np.float32)
    if X.ndim != 2:
        raise HTTPException(status_code=422, detail="features doit être une matrice 2D")

    threshold = _thresholds.get(req.model_name, settings.default_threshold)
    scores = model.score_samples(X).tolist()
    labels = [int(s >= threshold) for s in scores]

    return PredictResponse(
        scores=scores,
        labels=labels,
        threshold=threshold,
        model=req.model_name,
    )


@app.post("/threshold/{model_name}", tags=["configuration"])
def update_threshold(model_name: str, threshold: float) -> dict[str, str]:
    """Met à jour le seuil de classification d'un modèle."""
    if model_name not in _model_cache:
        raise HTTPException(status_code=404, detail=f"Modèle '{model_name}' non chargé")
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=422, detail="threshold ∈ [0, 1]")
    _thresholds[model_name] = threshold
    return {"message": f"Seuil de '{model_name}' mis à jour : {threshold}"}


def run() -> None:
    uvicorn.run(
        "anomaly_detection.serving.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
