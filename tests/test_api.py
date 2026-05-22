import json
import pytest
import numpy as np
from api.main import predict
import api.main as api_main

# On crée une fausse classe pour simuler le modèle LightGBM
class DummyModel:
    def predict(self, data):
        # Renvoie une fausse prédiction (0.5) pour chaque ligne reçue
        return np.array([0.5], dtype=np.float32)

# Cette fonction s'exécute AUTOMATIQUEMENT avant chaque test pour injecter le faux modèle
@pytest.fixture(autouse=True)
def setup_mock_model():
    api_main.model = DummyModel()
    if not api_main.expected_features:
        api_main.expected_features = ["NAME_CONTRACT_TYPE", "CODE_GENDER"]

def test_predict_with_valid_json():
    """Vérifie que l'API renvoie une prédiction valide avec un JSON correct."""
    valid_data = {feature: 0.0 for feature in api_main.expected_features}
    json_str = json.dumps(valid_data)

    response = predict(json_str)

    assert "Erreur" not in response
    assert float(response) == 0.5

def test_predict_with_invalid_json_format():
    """Vérifie le comportement de l'API face à un format de texte invalide."""
    bad_input = "ceci n'est pas un json"
    response = predict(bad_input)
    assert "Erreur : Le texte fourni n'est pas un JSON valide." in response

def test_predict_with_missing_variable():
    """Vérifie que l'API lève une erreur s'il manque une variable dans le JSON."""
    if len(api_main.expected_features) == 0:
        pytest.skip("La liste des variables est vide.")

    valid_data = {feature: 0.0 for feature in api_main.expected_features}
    missing_feature = api_main.expected_features[0]
    del valid_data[missing_feature]

    json_str = json.dumps(valid_data)
    response = predict(json_str)

    assert f"Erreur : Variable manquante : {missing_feature}" in response

def test_predict_with_invalid_type():
    """Vérifie que l'API lève une erreur si une variable n'est pas un nombre."""
    if not api_main.expected_features:
        pytest.skip("La liste des variables est vide.")

    valid_data = {feature: 0.0 for feature in api_main.expected_features}
    
    feature_to_break = api_main.expected_features[0]
    valid_data[feature_to_break] = "texte invalide"

    json_str = json.dumps(valid_data)
    response = predict(json_str)

    assert f"Erreur : La variable {feature_to_break} doit être un nombre." in response    