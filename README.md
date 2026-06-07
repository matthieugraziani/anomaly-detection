# Anomaly Detection

[![CI](https://github.com/matthieugraziani/anomaly-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/matthieugraziani/anomaly-detection/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-orange.svg)](https://mlflow.org/)

Pipeline complet de détection d'anomalies non supervisée, avec benchmark de modèles, API de prédiction et dashboard interactif.

---

## Fonctionnalités

- **3 modèles** : Isolation Forest, Autoencoder (PyTorch), LSTM-Autoencoder
- **Benchmark reproductible** : comparaison AUC-ROC, F1, Average Precision via MLflow
- **Feature engineering** : rolling statistics, lags, normalisation RobustScaler
- **API REST** (FastAPI) avec endpoint `/predict` et gestion du seuil par modèle
- **Dashboard** (Streamlit) : visualisation des scores, seuil interactif, table des anomalies
- **CI/CD** GitHub Actions : lint (Ruff + mypy), tests (pytest + couverture), benchmark manuel

---

## Architecture

```
Données (CSV / kagglehub / synthétique)
        │
        ▼
Feature engineering ──► Normalisation (RobustScaler)
        │
        ├──► Isolation Forest   (sklearn)
        ├──► Autoencoder        (PyTorch — reconstruction error)
        └──► LSTM-AE            (PyTorch — séquences temporelles)
                │
                ▼
        Benchmark + MLflow
                │
        ┌───────┴────────┐
        ▼                ▼
    FastAPI          Streamlit
   /predict          Dashboard
```

---

## Installation

```bash
git clone https://github.com/matthieugraziani/anomaly-detection.git
cd anomaly-detection
pip install -e ".[dev]"
```

### Données

Le projet utilise le dataset **Credit Card Fraud** (Kaggle) — 284 807 transactions, 492 fraudes (0.17%).

```bash
pip install kagglehub
```

Le téléchargement est automatique au premier lancement : si `data/raw/creditcard.csv` est absent,
`load_csv` appelle `kagglehub.dataset_download("mlg-ulb/creditcardfraud")` et copie le fichier.

Prérequis : fichier `~/.kaggle/kaggle.json` avec vos credentials API
(Kaggle → Account → API → *Create New Token*).

Ou utiliser le générateur de données synthétiques intégré (aucune clé requise) :

```bash
python -c "from anomaly_detection.data.loader import generate_synthetic; generate_synthetic()"
```

---

## Utilisation

### Notebooks

```
notebooks/
├── 01_eda.ipynb                  # Analyse exploratoire, distribution, corrélations
├── 02_feature_engineering.ipynb  # Rolling stats, lags, séquences LSTM
└── 03_benchmark_models.ipynb     # Entraînement, courbes ROC/PR, comparaison
```

### Benchmark en ligne de commande

```bash
python -m anomaly_detection.evaluation.benchmark \
    --data data/raw/creditcard.csv \
    --target Class \
    --experiment fraud-detection
```

Résultats disponibles dans l'interface MLflow :

```bash
mlflow ui --port 5000
```

### API FastAPI

```bash
uvicorn anomaly_detection.serving.api:app --reload --port 8000
# ou via le script installé :
ad-api
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.1, -1.2, 0.8, 0.3]], "model_name": "isolation_forest"}'
```

Endpoints disponibles : `GET /health`, `POST /predict`, `POST /threshold/{model_name}`.

### Dashboard Streamlit

```bash
streamlit run src/anomaly_detection/serving/dashboard.py
# ou via le script installé :
ad-dashboard
```

---

## Benchmark — résultats sur creditcard.csv

| Modèle           | AUC-ROC | Average Precision | F1 (seuil opt.) |
|------------------|---------|-------------------|-----------------|
| Isolation Forest | 0.947   | 0.312             | 0.271           |
| Autoencoder      | 0.962   | 0.401             | 0.318           |
| LSTM-AE          | 0.971   | 0.453             | 0.347           |

> Résultats reproductibles avec `seed=42` et les hyperparamètres par défaut de `settings.py`.

---

## Tests

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

La CI exécute automatiquement lint + tests sur chaque push vers `main` ou `develop`.
Le workflow `benchmark.yml` est déclenché manuellement (`workflow_dispatch`) avec le choix du dataset (`synthetic` ou `creditcard`).

---

## Structure du projet

```
anomaly-detection/
├── pyproject.toml
├── README.md
├── .github/workflows/
│   ├── ci.yml           # lint (Ruff, mypy) + tests (pytest)
│   └── benchmark.yml    # benchmark manuel avec upload des modèles
├── data/
│   └── raw/             # ignoré par git (creditcard.csv ou synthetic.csv)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_benchmark_models.ipynb
├── src/
│   └── anomaly_detection/
│       ├── config/
│       │   └── settings.py          # Settings dataclass (frozen)
│       ├── data/
│       │   ├── loader.py            # load_csv, download_kaggle_dataset, generate_synthetic
│       │   └── features.py          # rolling stats, lags, build_sequences, normalize
│       ├── models/
│       │   ├── base.py              # AnomalyDetector ABC (fit / score_samples / save / load)
│       │   ├── isolation_forest.py
│       │   ├── autoencoder.py
│       │   └── lstm_ae.py
│       ├── evaluation/
│       │   ├── metrics.py           # evaluate, find_best_threshold
│       │   └── benchmark.py         # pipeline complet + MLflow tracking
│       └── serving/
│           ├── api.py               # FastAPI — /predict, /health, /threshold
│           └── dashboard.py         # Streamlit — visualisation interactive
└── tests/
    ├── test_features.py
    ├── test_models.py
    └── test_api.py
```

---

## Licence

MIT
