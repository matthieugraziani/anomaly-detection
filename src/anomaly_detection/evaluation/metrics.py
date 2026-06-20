"""Métriques d'évaluation pour la détection d'anomalies."""

from __future__ import annotations

import logging
import numpy as np

from numpy.typing import NDArray
from sklearn.metrics import (average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

def evaluate(
    y_true: NDArray[np.int32],
    scores: NDArray[np.float32],
    threshold: float,
) -> dict[str, float]:
    """Calcule les métriques principales.

    Args:
        y_true: labels binaires (1 = anomalie, 0 = normal)
        scores: scores d'anomalie ∈ [0, 1]
        threshold: seuil de classification

    Returns:
        dict avec auc_roc, average_precision, f1, precision, recall
    """
    y_pred = (scores >= threshold).astype(int)

    return {
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(
            np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_pred == 1), 1)
        ),
        "recall": float(
            np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1)
        ),
        "threshold": threshold,
        "n_anomalies_detected": int(y_pred.sum()),
        "n_anomalies_true": int(y_true.sum()),
    }


def find_best_threshold(
    y_true: NDArray[np.int32],
    scores: NDArray[np.float32],
    metric: str = "f1",
) -> float:
    """Cherche le seuil optimal sur la courbe precision-recall.

    Args:
        y_true: labels binaires
        scores: scores d'anomalie
        metric: 'f1' (défaut) ou 'f0.5' (favorise la précision)

    Returns:
        Seuil optimal
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)

    beta = 1.0 if metric == "f1" else 0.5
    beta2 = beta ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        f_scores = np.where(
            (precisions[:-1] + recalls[:-1]) > 0,
            (1 + beta2) * precisions[:-1] * recalls[:-1] / (beta2 * precisions[:-1] + recalls[:-1]),
            0.0,
        )

    best_idx = int(np.argmax(f_scores))
    best_threshold = float(thresholds[best_idx])

    logger.info(
        "Seuil optimal (max %s) : %.4f → précision=%.3f, rappel=%.3f, f1=%.3f",
        metric,
        best_threshold,
        precisions[best_idx],
        recalls[best_idx],
        f_scores[best_idx],
    )
    return best_threshold
