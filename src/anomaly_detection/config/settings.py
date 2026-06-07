"""Configuration centralisée du projet (immuable)."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # Chemins
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")

    # Reproductibilité
    seed: int = 42

    # Feature engineering
    rolling_windows: tuple[int, ...] = (5, 10, 30)
    lag_periods: tuple[int, ...] = (1, 3, 7)

    # Isolation Forest
    if_n_estimators: int = 200
    if_contamination: float = 0.01
    if_max_features: float = 1.0

    # Autoencoder
    ae_hidden_dims: tuple[int, ...] = (64, 32, 16)
    ae_epochs: int = 50
    ae_batch_size: int = 256
    ae_lr: float = 1e-3
    ae_dropout: float = 0.1

    # LSTM-AE
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_seq_len: int = 10
    lstm_epochs: int = 30
    lstm_batch_size: int = 128
    lstm_lr: float = 1e-3

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    default_threshold: float = 0.5

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "anomaly-detection"


# Instance globale partagée
settings = Settings()
