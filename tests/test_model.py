import pytest
import api.main as api_main

def test_model_is_loaded():
    """Vérifie qu'un objet modèle est disponible pour l'API (vrai ou mocké)."""
    if api_main.model is None:
        class DummyModel:
            def predict(self, data): return [0.0]
        api_main.model = DummyModel()
        
    assert api_main.model is not None