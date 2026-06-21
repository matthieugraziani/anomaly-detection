"""LSTM Autoencoder pour séquences temporelles."""

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


class _LSTMAE(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, n_features, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        seq_len = x.size(1)

        h_last = h[-1].unsqueeze(1).repeat(1, seq_len, 1)

        out, _ = self.decoder(h_last)

        return torch.as_tensor(out)


class LSTMAEDetector(AnomalyDetector):
    """Détecteur basé sur un LSTM Autoencoder (erreur de reconstruction séquentielle)."""

    name = "lstm_ae"

    def __init__(
        self,
        seq_len: int = settings.lstm_seq_len,
        hidden_size: int = settings.lstm_hidden_size,
        num_layers: int = settings.lstm_num_layers,
        epochs: int = settings.lstm_epochs,
        batch_size: int = settings.lstm_batch_size,
        lr: float = settings.lstm_lr,
    ) -> None:
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._model: _LSTMAE | None = None
        self._scaler = MinMaxScaler()
        self._fitted = False
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_sequences(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        n = len(X) - self.seq_len + 1
        return np.stack([X[i : i + self.seq_len] for i in range(n)])

    def fit(self, X: NDArray[np.float32]) -> LSTMAEDetector:
        n_features = X.shape[1]
        seqs = self._build_sequences(X)
        self._model = _LSTMAE(n_features, self.hidden_size, self.num_layers).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        tensor = torch.tensor(seqs, dtype=torch.float32)
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
                logger.info("LSTM Epoch %d/%d — loss=%.6f",
                            epoch + 1,
                            self.epochs,
                            total_loss / len(loader))

        errors = self._seq_errors(seqs)
        self._scaler.fit(errors.reshape(-1, 1))
        self._fitted = True
        return self

    def _seq_errors(self, seqs: NDArray[np.float32]) -> NDArray[np.float32]:
        assert self._model is not None

        self._model.eval()

        with torch.no_grad():
            tensor = torch.tensor(
                seqs,
                dtype=torch.float32,
            ).to(self._device)

            recon = np.asarray(
                self._model(tensor).cpu().numpy(),
                dtype=np.float32,
            )

        errors = np.mean(
            (seqs - recon) ** 2,
            axis=(1, 2),
        )

        return np.asarray(errors, dtype=np.float32)

    def score_samples(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        if not self._fitted or self._model is None:
            raise RuntimeError("Appelez fit() avant score_samples()")
        # Padding pour retourner un score par sample (pas par séquence)
        seqs = self._build_sequences(X)
        seq_errors = self._seq_errors(seqs)
        # Les seq_len-1 premiers samples héritent du score de la première séquence
        pad = np.full(self.seq_len - 1, seq_errors[0], dtype=np.float32)
        errors = np.concatenate([pad, seq_errors])
        scores = np.asarray(
            self._scaler.transform(errors.reshape(-1, 1)).flatten(),
            dtype=np.float32,
        )

        return np.clip(
            scores,
            0.0,
            1.0,
        ).astype(np.float32)
