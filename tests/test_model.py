import numpy as np
import pytest
from api.main import model, expected_features

def test_model_is_loaded():
    """Vérifie que l'objet modèle MLflow n'est pas vide."""
    assert model is not None

def test_model_prediction_output_shape():
    """Vérifie que le modèle renvoie une prédiction avec la bonne structure."""
    if model is None or len(expected_features) == 0:
        pytest.skip("Le modèle ou les variables ne sont pas chargés.")
        
    # Création d'une fausse ligne de données avec le nombre exact de variables requises
    dummy_input = np.zeros((1, len(expected_features)), dtype=np.float32)
    
    # Appel de la fonction predict du modèle MLflow brut
    prediction = model.predict(dummy_input)
    
    # On vérifie qu'on obtient bien un résultat (un tableau ou une liste)
    assert prediction is not None
    assert len(prediction) == 1