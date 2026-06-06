"""Chargement et génération de données."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from anomaly_detection.config import settings

logger = logging.getLogger(__name__)


def load_csv(path: str | Path, target_col: str | None = None) -> tuple[pd.DataFrame, pd.Series | None]:
    """Charge un fichier CSV et sépare features / cible."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)
    logger.info("Chargé %s — %d lignes, %d colonnes", path.name, len(df), df.shape[1])

    y: pd.Series | None = None
    if target_col and target_col in df.columns:
        y = df.pop(target_col)
        logger.info("Cible '%s' extraite — %d positifs (%.2f%%)",
                    target_col, y.sum(), y.mean() * 100)

    return df, y


def load_parquet(path: str | Path, target_col: str | None = None) -> tuple[pd.DataFrame, pd.Series | None]:
    """Charge un fichier Parquet."""
    path = Path(path)
    df = pd.read_parquet(path)
    y: pd.Series | None = None
    if target_col and target_col in df.columns:
        y = df.pop(target_col)
    return df, y


def generate_synthetic(
    n_normal: int = 5000,
    n_anomaly: int = 100,
    n_features: int = 20,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Génère un dataset synthétique avec anomalies injectées.

    Les anomalies sont des points tirés d'une distribution à plus forte variance,
    simulant un comportement hors-norme.
    """
    rng = np.random.default_rng(settings.seed)

    # Points normaux (distribution gaussienne)
    X_normal = rng.standard_normal((n_normal, n_features))

    # Anomalies (variance plus élevée + décalage)
    X_anomaly = rng.standard_normal((n_anomaly, n_features)) * 3 + rng.uniform(-4, 4, n_features)

    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

    cols = [f"feature_{i:02d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    labels = pd.Series(y.astype(int), name="label")

    if save:
        out = settings.raw_dir / "synthetic.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([df, labels], axis=1).to_csv(out, index=False)
        logger.info("Dataset synthétique sauvegardé → %s", out)

    logger.info(
        "Dataset généré — %d normaux / %d anomalies (%.1f%%)",
        n_normal, n_anomaly, n_anomaly / (n_normal + n_anomaly) * 100,
    )
    return df, labels
