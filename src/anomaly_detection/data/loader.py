"""Chargement et génération de données."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import shutil

from anomaly_detection.config import settings
from pathlib import Path

logger = logging.getLogger(__name__)

_KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
_KAGGLE_FILENAME = "creditcard.csv"


# ---------------------------------------------------------------------------
# Téléchargement Kaggle
# ---------------------------------------------------------------------------

def download_kaggle_dataset(
    dest_dir: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Télécharge le dataset Credit Card Fraud depuis Kaggle via kagglehub.

    kagglehub télécharge dans son propre cache local et retourne le chemin.
    Le CSV est ensuite copié dans ``dest_dir`` (ex : data/raw/).

    Prérequis :
        - ``pip install kagglehub``
        - Fichier ``~/.kaggle/kaggle.json`` avec vos credentials API
          (ou variables d'env ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``)

    Args:
        dest_dir: Dossier de destination. Par défaut ``settings.raw_dir``.
        force: Retélécharge même si le fichier existe déjà dans dest_dir.

    Returns:
        Chemin vers le fichier CSV dans ``dest_dir``.

    Raises:
        OSError: Si kagglehub n'est pas installé.
        RuntimeError: Si le CSV est introuvable dans le cache après téléchargement.
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise OSError(
            "kagglehub n'est pas installé.\n"
            "  → pip install kagglehub\n"
            "  → Placez kaggle.json dans ~/.kaggle/\n"
            "  → Ou exportez KAGGLE_USERNAME et KAGGLE_KEY"
        ) from exc

    dest_dir = Path(dest_dir) if dest_dir else Path(settings.raw_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / _KAGGLE_FILENAME

    if target.exists() and not force:
        logger.info("Dataset déjà présent : %s", target)
        return target

    logger.info("Téléchargement '%s' via kagglehub…", _KAGGLE_DATASET)
    cache_dir = Path(kagglehub.dataset_download(_KAGGLE_DATASET))
    logger.info("Cache kagglehub : %s", cache_dir)

    candidates = list(cache_dir.rglob(_KAGGLE_FILENAME))
    if not candidates:
        raise RuntimeError(
            f"'{_KAGGLE_FILENAME}' introuvable dans le cache : {cache_dir}\n"
            f"Contenu : {list(cache_dir.iterdir())}"
        )

    shutil.copy2(candidates[0], target)
    logger.info("Dataset prêt : %s (%.1f Mo)", target, target.stat().st_size / 1e6)
    return target


# ---------------------------------------------------------------------------
# Chargement CSV / Parquet
# ---------------------------------------------------------------------------

def load_csv(
    path: str | Path,
    target_col: str | None = None,
    *,
    auto_download: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Charge un fichier CSV et sépare features / cible.

    Si le fichier est absent et que ``auto_download=True``, tente un
    téléchargement automatique depuis Kaggle avant de lever une erreur.
    """
    path = Path(path)

    if not path.exists():
        if auto_download and path.name == _KAGGLE_FILENAME:
            logger.warning("Fichier absent : %s — tentative de téléchargement…", path)
            try:
                path = download_kaggle_dataset(dest_dir=path.parent)
            except (OSError, RuntimeError) as exc:
                logger.error("Téléchargement impossible : %s", exc)
                raise FileNotFoundError(
                    f"Fichier introuvable et téléchargement échoué : {path}\n"
                    f"Détail : {exc}"
                ) from exc
        else:
            raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)
    logger.info("Chargé %s — %d lignes, %d colonnes", path.name, len(df), df.shape[1])

    y: pd.Series | None = None
    if target_col and target_col in df.columns:
        y = df.pop(target_col)
        logger.info(
            "Cible '%s' extraite — %d positifs (%.2f%%)",
            target_col, y.sum(), y.mean() * 100,
        )

    return df, y


def load_parquet(
    path: str | Path,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Charge un fichier Parquet."""
    path = Path(path)
    df = pd.read_parquet(path)
    y: pd.Series | None = None
    if target_col and target_col in df.columns:
        y = df.pop(target_col)
    return df, y


# ---------------------------------------------------------------------------
# Génération synthétique (fallback)
# ---------------------------------------------------------------------------

def generate_synthetic(
    n_normal: int = 5000,
    n_anomaly: int = 100,
    n_features: int = 20,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Génère un dataset synthétique avec anomalies injectées.

    Les anomalies sont tirées d'une distribution à plus forte variance,
    simulant un comportement hors-norme.
    """
    rng = np.random.default_rng(settings.seed)

    X_normal = rng.standard_normal((n_normal, n_features))
    X_anomaly = (
        rng.standard_normal((n_anomaly, n_features)) * 3
        + rng.uniform(-4, 4, n_features)
    )

    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

    cols = [f"feature_{i:02d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    labels = pd.Series(y.astype(int), name="label")

    if save:
        out = Path(settings.raw_dir) / "synthetic.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([df, labels], axis=1).to_csv(out, index=False)
        logger.info("Dataset synthétique sauvegardé → %s", out)

    logger.info(
        "Dataset généré — %d normaux / %d anomalies (%.1f%%)",
        n_normal, n_anomaly, n_anomaly / (n_normal + n_anomaly) * 100,
    )
    return df, labels
