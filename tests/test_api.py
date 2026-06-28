import io
import pytest
import numpy as np
from fastapi.testclient import TestClient

from api.main import app
import api.main as api_main


# --- 1. MOCK DE L'ENVIRONNEMENT ---

class DummyInput:
    @property
    def name(self):
        return "float_input"


class DummyModel:
    def get_inputs(self):
        return [DummyInput()]

    def run(self, output_names, feed_dict):
        X = list(feed_dict.values())[0]
        predictions = np.full(len(X), 0.5)
        return [predictions]


@pytest.fixture(autouse=True)
def setup_mock_env():
    api_main.model = DummyModel()
    api_main.expected_features = [
        "CREDIT_INCOME_RATIO",
        "AMT_CREDIT",
        "CODE_GENDER_F"
    ]

    # On désactive les logs BDD pendant les tests
    api_main.DATABASE_URL = None


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
    """Vérifie que le feature engineering et la prédiction fonctionnent avec un CSV valide."""
    csv_content = (
        "AMT_CREDIT,AMT_INCOME_TOTAL,AMT_ANNUITY,DAYS_BIRTH,DAYS_EMPLOYED,CODE_GENDER\n"
        "2000,1000,50,-10000,-500,F\n"
    )

    file_like = io.BytesIO(csv_content.encode("utf-8"))

    response = client.post(
        "/predict",
        files={"file": ("donnees_client.csv", file_like, "text/csv")}
    )

    if response.status_code != 200:
        print(f"\nDÉTAIL DU CRASH API : {response.text}")

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 1
    assert data["predictions"][0] == 0.5


def test_predict_rejects_missing_columns():
    """Vérifie que l'API bloque s'il manque des colonnes obligatoires."""
    csv_content = "AMT_CREDIT,CODE_GENDER\n2000,F\n"
    file_like = io.BytesIO(csv_content.encode("utf-8"))

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