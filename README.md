---
title: Mlops Api
emoji: 🚀
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# API de scoring crédit - Projet MLOps OpenClassrooms

## Présentation

Ce projet déploie une API de scoring crédit permettant de prédire le risque de défaut de paiement d’un client à partir de données financières et personnelles.

Il s’appuie sur un projet précédent de modélisation, construit à partir de la fusion de six datasets.

Le modèle est exposé via une API FastAPI, déployée sur Hugging Face, avec un frontend Gradio.

Application déployée :

```text
https://huggingface.co/spaces/EthanChmt/mlops-scoring-front
```

Le projet repose sur trois espaces Hugging Face :

- un dépôt modèle ;
- un backend API ;
- un frontend Gradio.

## Architecture du dépôt

```text
.
├── api/
│   ├── main.py
│   └── expected_features.json
│
├── tests/
│   ├── test_api.py
│   └── test_model.py
│
├── monitoring/
│   ├── data_drift_analysis.ipynb
│   └── Optimisation_report.md
│
├── reports/
│   └── data_drift_report.html
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── pyproject.toml
├── poetry.lock
└── README.md
```

## API FastAPI

L’application principale est une API FastAPI.

Elle expose deux endpoints :

| Endpoint | Méthode | Rôle |
| --- | --- | --- |
| `/health` | GET | Vérifie que l’API est disponible |
| `/predict` | POST | Reçoit un fichier CSV et retourne la classe prédite |

Le endpoint `/predict` attend un fichier CSV contenant les colonnes nécessaires au scoring et au feature engineering.

Exemple de réponse :

```json
{
  "predictions": [0]
}
```

La sortie correspond à la classe prédite par le modèle.

## Modèle et optimisation

Le modèle initial était au format `.pkl`.

Une optimisation post-déploiement a été réalisée en convertissant le modèle au format ONNX et en utilisant ONNX Runtime pour l’inférence.

Le modèle optimisé est chargé depuis Hugging Face au démarrage de l’API et exécuté sur CPU avec `CPUExecutionProvider`.

Le GPU n’a pas été retenu, car les performances obtenues sur CPU sont suffisantes pour cette API.

Les résultats détaillés de l’optimisation sont disponibles dans :

```text
monitoring/Optimisation_report.md
```

## Monitoring et Data Drift

L’API enregistre les informations utiles au suivi des prédictions dans une base PostgreSQL/Neon :

- données d’entrée ;
- prédiction ;
- statut de la requête ;
- temps d’exécution ;
- date de prédiction.

Une analyse du Data Drift est réalisée avec Evidently à partir des données de référence et des données de production récupérées depuis les logs de l’API.

Le notebook applique le même feature engineering que l’API avant de comparer les distributions des features utilisées par le modèle.

Fichiers associés :

```text
monitoring/data_drift_analysis.ipynb
reports/data_drift_report.html
```

## Tests automatisés

Les tests sont écrits avec Pytest.

Ils vérifient notamment :

- le fonctionnement de `/health` ;
- le fonctionnement de `/predict` ;
- le rejet des fichiers non CSV ;
- le rejet des requêtes avec colonnes manquantes.

Commande PowerShell, ou équivalent Linux/macOS :

```powershell
poetry run pytest tests -v
```

Sans Poetry :

```powershell
python -m pytest tests -v
```

## Installation locale

Le projet utilise Poetry pour la gestion des dépendances.

Commandes PowerShell, ou équivalent Linux/macOS :

```powershell
poetry install
```

```powershell
poetry run uvicorn api.main:app --host 0.0.0.0 --port 7860
```

Documentation interactive FastAPI :

```text
http://127.0.0.1:7860/docs
```

## Variables d’environnement

Exemple de fichier `.env` :

```env
DATABASE_URL=postgresql://user:password@host:port/database
HF_MODEL_REPO=EthanChmt/scoring_api_OC
HF_MODEL_FILENAME=model.onnx
```

`DATABASE_URL` active le logging en base PostgreSQL/Neon.

`HF_MODEL_REPO` et `HF_MODEL_FILENAME` permettent de charger le modèle ONNX depuis Hugging Face.

## Docker

Commandes PowerShell, ou équivalent Linux/macOS :

```powershell
docker build -t mlops-scoring-api .
```

```powershell
docker run -p 7860:7860 mlops-scoring-api
```

L’API est ensuite disponible ici :

```text
http://127.0.0.1:7860/docs
```

## Exemple d’appel API local

Commande PowerShell, ou équivalent Linux/macOS :

```powershell
curl.exe -X POST -F "file=@test_api_input.csv" http://127.0.0.1:7860/predict
```

## CI/CD

Le pipeline CI/CD est défini dans :

```text
.github/workflows/ci.yml
```

Il lance les tests et permet le déploiement vers Hugging Face.

La branche principale de développement est :

```text
develop
```

Des branches de travail ont été utilisées pendant le projet :

```text
feature/api-docker
feature/cd
feature/monitoring
```

Les développements validés ont vocation à converger vers `main`.

## Frontend Gradio

Un frontend Gradio permet d’utiliser le modèle sans appeler directement les endpoints FastAPI.

Frontend déployé :

```text
https://huggingface.co/spaces/EthanChmt/mlops-scoring-front
```

## Auteur

Projet réalisé par Ethan Chaumeret dans le cadre du parcours OpenClassrooms MLOps.