"""Autoencoder PyTorch pour la détection d'anomalies."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from anomaly_detection.config import settings
from anomaly_detection.models.base import AnomalyDetector

logger = logging.getLogger(__name__)


class _AE(nn.Module):
    def __init__(self, n_features: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        dims = [n_features, *hidden_dims]
        encoder_layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            encoder_layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.encoder = nn.Sequential(*encoder_layers)

        rdims = list(reversed(dims))
        decoder_layers: list[nn.Module] = []
        for i in range(len(rdims) - 1):
            decoder_layers += [nn.Linear(rdims[i], rdims[i + 1])]
            if i < len(rdims) - 2:
                decoder_layers += [nn.ReLU(), nn.Dropout(dropout)]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return torch.as_tensor(decoded)


class AutoencoderDetector(AnomalyDetector):
    """Détecteur basé sur un Autoencoder dense (reconstruction error)."""

    name = "autoencoder"

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = settings.ae_hidden_dims,
        epochs: int = settings.ae_epochs,
        batch_size: int = settings.ae_batch_size,
        lr: float = settings.ae_lr,
        dropout: float = settings.ae_dropout,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self._model: _AE | None = None
        self._scaler = MinMaxScaler()
        self._fitted = False
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: NDArray[np.float32]) -> "AutoencoderDetector":
        n_features = X.shape[1]
        self._model = _AE(n_features, self.hidden_dims, self.dropout).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        tensor = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor), batch_size=self.batch_size, shuffle=True)

        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()
                loss = criterion(self._model(batch), batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info("Epoch %d/%d — loss=%.6f", epoch + 1, self.epochs, total_loss / len(loader))

        # Calibrer le scaler sur les erreurs de reconstruction
        errors = self._reconstruction_errors(X)
        self._scaler.fit(errors.reshape(-1, 1))
        self._fitted = True
        return self

    def _reconstruction_errors(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        assert self._model is not None

        self._model.eval()

        with torch.no_grad():
            tensor = torch.tensor(
                X,
                dtype=torch.float32,
            ).to(self._device)

            recon = np.asarray(
                self._model(tensor).cpu().numpy(),
                dtype=np.float32,
            )

        errors = np.mean(
            (X - recon) ** 2,
            axis=1,
        )

        return np.asarray(
            errors,
            dtype=np.float32,
        )

    def score_samples(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        if not self._fitted or self._model is None:
            raise RuntimeError("Appelez fit() avant score_samples()")
        errors = self._reconstruction_errors(X)
        scores = np.asarray(
            self._scaler.transform(errors.reshape(-1, 1)).flatten(),
            dtype=np.float32,
        )

        return np.clip(
            scores,
            0.0,
            1.0,
        ).astype(np.float32)
