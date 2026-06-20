"""Tests des modèles de détection d'anomalies."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detection.models import ( AutoencoderDetector,
                                     IsolationForestDetector,
                                     LSTMAEDetector,
                                     )


@pytest.fixture
def normal_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((300, 10)).astype(np.float32)


@pytest.fixture
def test_data() -> np.ndarray:
    rng = np.random.default_rng(0)
    normal = rng.standard_normal((90, 10))
    anomalies = rng.standard_normal((10, 10)) * 5 + 10  # bien séparées
    return np.vstack([normal, anomalies]).astype(np.float32)


class TestIsolationForest:
    def test_fit_returns_self(self, normal_data: np.ndarray) -> None:
        model = IsolationForestDetector(n_estimators=10)
        result = model.fit(normal_data)
        assert result is model

    def test_scores_shape(self, normal_data: np.ndarray, test_data: np.ndarray) -> None:
        model = IsolationForestDetector(n_estimators=10).fit(normal_data)
        scores = model.score_samples(test_data)
        assert scores.shape == (len(test_data),)

    def test_scores_range(self, normal_data: np.ndarray, test_data: np.ndarray) -> None:
        model = IsolationForestDetector(n_estimators=10).fit(normal_data)
        scores = model.score_samples(test_data)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_not_fitted_raises(self, test_data: np.ndarray) -> None:
        model = IsolationForestDetector()
        with pytest.raises(RuntimeError):
            model.score_samples(test_data)

    def test_anomalies_higher_score(self, normal_data: np.ndarray) -> None:
        """Les anomalies doivent avoir un score moyen plus élevé que les normaux."""
        rng = np.random.default_rng(1)
        anomalies = (rng.standard_normal((20, 10)) * 5 + 10).astype(np.float32)
        model = IsolationForestDetector(n_estimators=50).fit(normal_data)

        score_normal = model.score_samples(normal_data[:50]).mean()
        score_anomaly = model.score_samples(anomalies).mean()
        assert score_anomaly > score_normal


class TestAutoencoder:
    def test_fit_and_score(self, normal_data: np.ndarray, test_data: np.ndarray) -> None:
        model = AutoencoderDetector(epochs=2, batch_size=64).fit(normal_data)
        scores = model.score_samples(test_data)
        assert scores.shape == (len(test_data),)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)


class TestLSTMAE:
    def test_fit_and_score(self, normal_data: np.ndarray) -> None:
        model = LSTMAEDetector(seq_len=5, epochs=2, batch_size=32).fit(normal_data)
        scores = model.score_samples(normal_data)
        # Le padding ramène à la bonne longueur
        assert scores.shape == (len(normal_data),)
