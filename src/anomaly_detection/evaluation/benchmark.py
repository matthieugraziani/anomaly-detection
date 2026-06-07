"""Benchmark multi-modèles avec tracking MLflow.

Usage :
    python -m anomaly_detection.evaluation.benchmark \
        --data data/raw/creditcard.csv \
        --target Class
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from anomaly_detection.config import settings
from anomaly_detection.data.features import normalize
from anomaly_detection.data.loader import load_csv
from anomaly_detection.evaluation.metrics import evaluate, find_best_threshold
from anomaly_detection.models import (
    AnomalyDetector,
    AutoencoderDetector,
    IsolationForestDetector,
    LSTMAEDetector,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_benchmark(
    data_path: str | Path,
    target_col: str,
    experiment_name: str = settings.mlflow_experiment,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Entraîne et évalue les 3 modèles, log les résultats dans MLflow.

    Returns:
        DataFrame de comparaison des résultats
    """
    # Chargement
    df, y = load_csv(data_path, target_col)
    if y is None:
        raise ValueError(f"Colonne cible '{target_col}' introuvable")

    X = df.select_dtypes(include="number").values.astype(np.float32)
    y_np = y.values.astype(np.int32)

    # Split train/test (l'entraînement se fait sur les normaux uniquement)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_np, test_size=test_size, random_state=settings.seed, stratify=y_np
    )
    X_train_normal = X_train[y_train == 0]

    # Normalisation
    X_train_scaled, X_test_scaled, _ = normalize(X_train_normal, X_test)
    assert X_test_scaled is not None

    # Modèles à benchmarker
    detectors: list[AnomalyDetector] = [
        IsolationForestDetector(),
        AutoencoderDetector(),
        LSTMAEDetector(),
    ]

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    results = []
    for detector in detectors:
        logger.info("=== %s ===", detector.name)
        with mlflow.start_run(run_name=detector.name):
            # Entraînement
            detector.fit(X_train_scaled)

            # Scoring
            scores = detector.score_samples(X_test_scaled)
            threshold = find_best_threshold(y_test, scores)
            metrics = evaluate(y_test, scores, threshold)

            # Log MLflow
            mlflow.log_params({
                "model": detector.name,
                "n_train": len(X_train_normal),
                "n_test": len(X_test),
                "threshold": threshold,
            })
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})

            # Sauvegarde du modèle
            model_path = Path("models") / f"{detector.name}.pkl"
            detector.save(model_path)
            mlflow.log_artifact(str(model_path))

            results.append({"model": detector.name, **metrics})
            logger.info(
                "AUC-ROC=%.4f, AP=%.4f, F1=%.4f",
                metrics["auc_roc"],
                metrics["average_precision"],
                metrics["f1"],
            )

    df_results = pd.DataFrame(results).set_index("model")
    print("\n=== Résultats benchmark ===")
    print(df_results[["auc_roc", "average_precision", "f1", "precision", "recall"]].to_string())
    return df_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark de détection d'anomalies")
    parser.add_argument("--data", required=True, help="Chemin vers le CSV")
    parser.add_argument("--target", required=True, help="Nom de la colonne cible")
    parser.add_argument("--experiment", default=settings.mlflow_experiment)
    args = parser.parse_args()

    run_benchmark(args.data, args.target, args.experiment)


if __name__ == "__main__":
    main()
