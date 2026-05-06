import mlflow.pyfunc
import numpy as np
import logging
import warnings
from src.model_utils import load_data

# Configuration des logs pour une sortie propre
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def make_predictions_with_champions():
    """
    Charge les 3 'champions' depuis le Model Registry et effectue des prédictions.
    
    """
    try:
        # 1. Chargement des données (Numpy arrays)
        X_full, _ = load_data()
        
        # Sélection d'un échantillon pour la démonstration
        sample_data = X_full[0:5]
        logger.info(f"Données de test chargées. Taille du sample : {sample_data.shape}")

        # 2. Liste des champions (Noms exacts enregistrés dans le notebook)
        champions = [
            "Champion_Logistic_Regression",
            "Champion_XGBoost",
            "Champion_LightGBM"
        ]
        
        for model_name in champions:
            print("-" * 50)
            try:
                # On récupère la version 1 du modèle dans le Registry
                model_uri = f"models:/{model_name}/1"
                logger.info(f"Chargement du modèle : {model_name}...")
                
                # Chargement du modèle (inclut les poids/paramètres appris)
                model = mlflow.pyfunc.load_model(model_uri)
                
                # Exécution de la prédiction
                preds = model.predict(sample_data)
                
                print(f"RÉSULTAT pour {model_name}:")
                print(f" -> Prédictions de classes : {preds}")
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {model_name} : {e}")

    except Exception as e:
        logger.error(f"Erreur lors du chargement des données : {e}")

if __name__ == "__main__":
    make_predictions_with_champions()