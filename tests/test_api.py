"""Tests de l'API FastAPI (sans modèle chargé — mock)."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detection.models import IsolationForestDetector
from anomaly_detection.serving import api as api_module
from anomaly_detection.serving.api import app
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def mock_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Injecte un modèle entraîné rapidement dans le cache de l'API."""
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((200, 5)).astype(np.float32)
    model = IsolationForestDetector(n_estimators=10).fit(X_train)

    monkeypatch.setitem(api_module._model_cache, "isolation_forest", model)
    monkeypatch.setitem(api_module._thresholds, "isolation_forest", 0.5)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "isolation_forest" in data["models_loaded"]


def test_predict_valid(client: TestClient) -> None:
    payload = {"features": [[0.1, -0.5, 1.2, 0.3, -0.8]], "model_name": "isolation_forest"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["scores"]) == 1
    assert len(data["labels"]) == 1
    assert 0.0 <= data["scores"][0] <= 1.0
    assert data["labels"][0] in (0, 1)


def test_predict_batch(client: TestClient) -> None:
    features = [[float(i) * 0.1, -float(i) * 0.2, 0.5, 0.1, -0.3] for i in range(10)]
    payload = {"features": features, "model_name": "isolation_forest"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert len(response.json()["scores"]) == 10


def test_predict_unknown_model(client: TestClient) -> None:
    payload = {"features": [[0.1, 0.2, 0.3, 0.4, 0.5]], "model_name": "inexistant"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 404


def test_update_threshold(client: TestClient) -> None:
    response = client.post("/threshold/isolation_forest?threshold=0.7")
    assert response.status_code == 200


def test_update_threshold_invalid(client: TestClient) -> None:
    response = client.post("/threshold/isolation_forest?threshold=1.5")
    assert response.status_code == 422
