import io
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Import de l'application FastAPI
from api.main import app
import api.main as api_main

# --- 1. MOCK DE L'ENVIRONNEMENT ---
# On crée une fausse entrée ONNX
class DummyInput:
    @property
    def name(self):
        return "float_input"

# Le nouveau faux modèle qui parle couramment le langage ONNX
class DummyModel:
    def get_inputs(self):
        return [DummyInput()]

    def run(self, output_names, feed_dict):
        # On simule le moteur ONNX : on récupère les données envoyées
        X = list(feed_dict.values())[0]
        # On renvoie un tableau Numpy de 0.5 (1 prédiction bidon par client)
        predictions = np.full(len(X), 0.5)
        return [predictions]

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
    if response.status_code != 200:
        print(f"\n🚨 DÉTAIL DU CRASH API : {response.text}")
    assert response.status_code == 200

def test_predict_rejects_missing_columns():
    """Vérifie que l'API bloque strictement s'il manque des données pour l'EDA."""
    csv_content = "AMT_CREDIT,CODE_GENDER\n2000,F\n"
    file_like = io.BytesIO(csv_content.encode('utf-8'))

    response = client.post(
        "/predict",
        files={"file": ("donnees_client.csv", file_like, "text/csv")}
    )

    assert response.status_code == 400
    assert "Colonnes manquantes détectées" in response.json()["detail"]

def test_predict_rejects_invalid_file_type():
    """Vérifie le rejet des fichiers non-CSV."""
    file_like = io.BytesIO(b"Ceci est un fichier texte")

    response = client.post(
        "/predict",
        files={"file": ("document.txt", file_like, "text/plain")}
    )

    assert response.status_code == 400
    assert "Fichier CSV requis." in response.json()["detail"]