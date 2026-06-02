import io
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Import de l'application FastAPI
from api.main import app
import api.main as api_main

# --- 1. MOCK DE L'ENVIRONNEMENT ---
# On crée un faux modèle pour ne pas dépendre du vrai .pkl lourd pendant les tests
class DummyModel:
    def predict(self, df):
        # Renvoie une probabilité de 0.5 pour chaque ligne
        return np.full(len(df), 0.5)

# Cette fonction s'exécute automatiquement avant chaque test
@pytest.fixture(autouse=True)
def setup_mock_env():
    api_main.model = DummyModel()
    # On simule ce que devrait contenir expected_features.json après ton EDA
    api_main.expected_features = ['CREDIT_INCOME_RATIO', 'AMT_CREDIT', 'CODE_GENDER_F']

client = TestClient(app)

# --- 2. TESTS DES ROUTES ---

def test_health_check_ok():
    """Vérifie que l'API est bien en ligne."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_health_check_fails_if_no_model():
    """Vérifie que l'API signale si le modèle est indisponible."""
    api_main.model = None
    response = client.get("/health")
    assert response.status_code == 503

def test_predict_with_valid_csv():
    """Vérifie que le Feature Engineering et la prédiction fonctionnent avec un CSV parfait."""
    # Fichier brut avec TOUTES les colonnes exigées par l'API
    csv_content = "AMT_CREDIT,AMT_INCOME_TOTAL,AMT_ANNUITY,DAYS_BIRTH,DAYS_EMPLOYED,CODE_GENDER\n2000,1000,50,-10000,-500,F\n"
    file_like = io.BytesIO(csv_content.encode('utf-8'))

    response = client.post(
        "/predict",
        files={"file": ("donnees_client.csv", file_like, "text/csv")}
    )

    assert response.status_code == 200


def test_predict_rejects_missing_columns():
    """Vérifie que l'API bloque strictement s'il manque des données pour l'EDA."""
    # Il manque AMT_INCOME_TOTAL et d'autres.
    csv_content = "AMT_CREDIT,CODE_GENDER\n2000,F\n"
    file_like = io.BytesIO(csv_content.encode('utf-8'))

    response = client.post(
        "/predict",
        files={"file": ("donnees_client.csv", file_like, "text/csv")}
    )

    # L'API doit bloquer avec notre erreur 400
    assert response.status_code == 400
    # On vérifie la nouvelle phrase d'erreur exacte de l'API
    assert "Fichier invalide. Il manque ces colonnes essentielles" in response.json()["detail"]
def test_predict_rejects_invalid_file_type():
    """Vérifie le rejet des fichiers non-CSV."""
    file_like = io.BytesIO(b"Ceci est un fichier texte")
    
    response = client.post(
        "/predict",
        files={"file": ("document.txt", file_like, "text/plain")}
    )
    
    assert response.status_code == 400
    assert "Veuillez fournir un fichier CSV." in response.json()["detail"]