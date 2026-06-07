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
- **Feature engineering** : rolling statistics, lags, encodage temporel
- **API REST** (FastAPI) avec endpoint `/predict`
- **Dashboard** (Streamlit) : visualisation des scores, seuil interactif, alertes
- **CI/CD** GitHub Actions : lint, tests, couverture

---

## Architecture

```
Données (CSV / API / SQL)
        │
        ▼
Feature engineering ──► Normalisation
        │
        ├──► Isolation Forest
        ├──► Autoencoder (PyTorch)
        └──► LSTM-AE (PyTorch)
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
git clone https://github.com/youruser/anomaly-detection.git
cd anomaly-detection
pip install -e ".[dev]"
```

### Données de démonstration

Le projet utilise le dataset **Credit Card Fraud** (Kaggle) — 284 807 transactions, 492 fraudes.

```bash
# Télécharger via kaggle CLI
kaggle datasets download mlg-ulb/creditcardfraud -p data/raw/ --unzip
```

Ou utiliser le générateur de données synthétiques intégré :

```bash
python -c "from anomaly_detection.data.loader import generate_synthetic; generate_synthetic()"
```

---

## Utilisation

### Entraînement et benchmark

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
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.1, -1.2, 0.8, ...]]}'
```

### Dashboard Streamlit

```bash
streamlit run src/anomaly_detection/serving/dashboard.py
```

---

## Benchmark — résultats sur creditcard.csv

| Modèle           | AUC-ROC | Average Precision | F1 (seuil opt.) |
|------------------|---------|-------------------|-----------------|
| Isolation Forest | 0.947   | 0.312             | 0.271           |
| Autoencoder      | 0.962   | 0.401             | 0.318           |
| LSTM-AE          | 0.971   | 0.453             | 0.347           |

> Résultats reproductibles avec `seed=42` et les hyperparamètres par défaut.

---

## Tests

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Structure du projet

```
anomaly-detection/
├── pyproject.toml
├── README.md
├── .github/workflows/
│   ├── ci.yml
│   └── benchmark.yml
├── data/raw/              # ignoré par git
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_benchmark_models.ipynb
├── src/
│   ├── config/settings.py
│   ├── data/{loader,features}.py
│   ├── models/{base,isolation_forest,autoencoder,lstm_ae}.py
│   ├── evaluation/{metrics,benchmark}.py
│   └── serving/{api,dashboard}.py
└── tests/
```

---

## Licence

MIT
