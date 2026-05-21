import json
import numpy as np
import pytest
from api.main import predict, expected_features, model

def test_expected_features_loaded():
    """Vérifie que la liste des variables est bien chargée au démarrage."""
    assert expected_features is not None
    assert len(expected_features) > 0
    # On s'assure qu'on a bien nos variables principales (ex: TARGET ou équivalent)
    assert isinstance(expected_features, list)

def test_predict_with_valid_json():
    """Vérifie que l'API renvoie une prédiction valide avec un JSON correct."""
    # On simule un dictionnaire avec toutes les variables requises initialisées à 0.0
    valid_data = {feature: 0.0 for feature in expected_features}
    json_str = json.dumps(valid_data)
    
    response = predict(json_str)
    
    # La réponse ne doit pas être un message d'erreur
    assert "Erreur" not in response
    # La réponse doit pouvoir être convertie en nombre flottant (le score)
    try:
        float(response)
    except ValueError:
        pytest.fail(f"La réponse de l'API n'est pas un score numérique : {response}")

def test_predict_with_invalid_json_format():
    """Vérifie le comportement de l'API face à un format de texte invalide."""
    bad_input = "ceci n'est pas un json"
    response = predict(bad_input)
    assert "Erreur : Le texte fourni n'est pas un JSON valide." in response

def test_predict_with_missing_variable():
    """Vérifie que l'API lève une erreur s'il manque une variable dans le JSON."""
    if len(expected_features) == 0:
        pytest.skip("La liste des variables est vide.")
        
    valid_data = {feature: 0.0 for feature in expected_features}
    # On supprime volontairement la toute première variable requise
    missing_feature = expected_features[0]
    del valid_data[missing_feature]
    
    json_str = json.dumps(valid_data)
    response = predict(json_str)
    
    assert f"Erreur : Variable manquante : {missing_feature}" in response