"""Dashboard Streamlit pour la visualisation des anomalies détectées."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from anomaly_detection.config import settings
from anomaly_detection.data.loader import load_csv
from anomaly_detection.evaluation.metrics import evaluate, find_best_threshold
from anomaly_detection.models import ( AnomalyDetector, 
                                     AutoencoderDetector,
                                     IsolationForestDetector,
                                     )

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def load_model(name: str) -> AnomalyDetector | None:
    path = Path("models") / f"{name}.pkl"
    if path.exists():
        return AnomalyDetector.load(path)
    return None


@st.cache_data
def load_data(path: str | None, target_col: str) -> tuple[pd.DataFrame, pd.Series | None]:
    if not path:
        raise ValueError("Aucun fichier CSV fourni. Renseignez un chemin dans la barre latérale.")

    if not Path(path).exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    return load_csv(path, target_col)


def run() -> None:
    st.title("Détection d'anomalies")
    st.caption("Pipeline : Isolation Forest · Autoencoder · LSTM-AE")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")
        data_path = st.text_input("Chemin CSV", value="data/raw/creditcard.csv")
        target_col = st.text_input("Colonne cible", value="Class")

        model_name = st.selectbox(
            "Modèle",
            ["isolation_forest", "autoencoder", "lstm_ae"],
            format_func=lambda x: {
                "isolation_forest": "Isolation Forest",
                "autoencoder": "Autoencoder",
                "lstm_ae": "LSTM-AE",
            }[x],
        )
        threshold = st.slider("Seuil de détection", 0.0, 1.0, settings.default_threshold, 0.01)
        train_mode = st.checkbox("Entraîner un nouveau modèle", value=False)

    # ── Chargement données ────────────────────────────────────────────────────
    df, y = load_data(data_path or None, target_col)
    X = df.select_dtypes(include="number").values.astype(np.float32)

    st.subheader("Aperçu des données")
    col1, col2, col3 = st.columns(3)
    col1.metric("Échantillons", f"{len(df):,}")
    col2.metric("Features", df.shape[1])
    if y is not None:
        col3.metric("Anomalies vraies", f"{y.sum():,} ({y.mean() * 100:.2f}%)")

    # ── Entraînement / Chargement ─────────────────────────────────────────────
    model: AnomalyDetector | None = None

    if train_mode:
        if st.button("Lancer l'entraînement"):
            with st.spinner(f"Entraînement {model_name}…"):
                X_normal = X[y.values == 0] if y is not None else X
                if model_name == "isolation_forest":
                    model = IsolationForestDetector().fit(X_normal)
                else:
                    model = AutoencoderDetector().fit(X_normal)
                model.save(Path("models") / f"{model_name}.pkl")
                st.success("Modèle entraîné et sauvegardé")
                st.cache_resource.clear()
    else:
        model = load_model(model_name)
        if model is None:
            st.warning(
                f"Aucun modèle '{model_name}' trouvé. "
                "Activer 'Entraîner un nouveau modèle'."
            )

    # ── Prédictions ───────────────────────────────────────────────────────────
    if model is not None:
        scores = model.score_samples(X)

        if y is not None:
            auto_threshold = find_best_threshold(y.values.astype(np.int32), scores)
            col_t1, col_t2 = st.columns(2)
            col_t1.metric("Seuil manuel", f"{threshold:.3f}")
            col_t2.metric("Seuil optimal (F1)", f"{auto_threshold:.3f}")

            metrics = evaluate(y.values.astype(np.int32), scores, threshold)

            st.subheader("Métriques")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
            m2.metric("Average Precision", f"{metrics['average_precision']:.4f}")
            m3.metric("F1", f"{metrics['f1']:.4f}")
            m4.metric("Anomalies détectées", metrics["n_anomalies_detected"])

        labels = (scores >= threshold).astype(int)

        # ── Graphiques ────────────────────────────────────────────────────────
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.subheader("Distribution des scores")
            fig = go.Figure()
            if y is not None:
                fig.add_trace(go.Histogram(
                    x=scores[y.values == 0], name="Normal", opacity=0.7,
                    marker_color="#1D9E75", nbinsx=50,
                ))
                fig.add_trace(go.Histogram(
                    x=scores[y.values == 1], name="Anomalie", opacity=0.7,
                    marker_color="#D85A30", nbinsx=50,
                ))
            else:
                fig.add_trace(go.Histogram(x=scores, nbinsx=50, marker_color="#378ADD"))
            fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="seuil")
            fig.update_layout(barmode="overlay", height=350, margin=dict(t=10, b=10))
            st.plotly_chart(fig, width='stretch')

        with col_g2:
            st.subheader("Scores dans le temps")
            idx = np.arange(len(scores))
            sample = min(len(scores), 2000)
            idx_s = np.random.choice(len(scores), sample, replace=False)
            colors = ["#D85A30" if labels[i] else "#1D9E75" for i in idx_s]
            fig2 = go.Figure(go.Scatter(
                x=idx[idx_s], y=scores[idx_s], mode="markers",
                marker=dict(color=colors, size=3, opacity=0.6),
            ))
            fig2.add_hline(y=threshold, line_dash="dash", line_color="red")
            fig2.update_layout(height=350, margin=dict(t=10, b=10),
                               yaxis_title="Score d'anomalie")
            st.plotly_chart(fig2, width='stretch')

        # ── Table des anomalies détectées ─────────────────────────────────────
        st.subheader("Échantillons détectés comme anomalies")
        anomaly_idx = np.where(labels == 1)[0]
        if len(anomaly_idx) > 0:
            df_anomalies = df.iloc[anomaly_idx].copy()
            df_anomalies["score"] = scores[anomaly_idx]
            df_anomalies = df_anomalies.sort_values("score", ascending=False)
            st.dataframe(df_anomalies.head(50), width='stretch')
        else:
            st.info("Aucune anomalie détectée avec ce seuil.")


if __name__ == "__main__":
    run()
