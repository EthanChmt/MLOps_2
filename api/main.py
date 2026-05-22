import os
import sys
import json
import numpy as np
import gradio as gr
import logging
import mlflow.pyfunc

# 1. Gestion des chemins absolus (CRITIQUE POUR LES IMPORTS)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
os.chdir(root_path)

# 2. Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 3. Définition des chemins de fichiers de l'API
base_dir = os.path.abspath(os.path.dirname(__file__))
features_path = os.path.join(base_dir, "expected_features.json")
LOCAL_MODEL_PATH = os.path.join(base_dir, "model_files")

model = None
expected_features = []

# 4. Chargement du fichier des variables requises
try:
    with open(features_path, 'r') as f:
        expected_features = json.load(f)
    logger.info("Configuration des variables chargée avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement de expected_features.json : {e}")

# 5. CHARGEMENT DU MODÈLE
logger.info("Chargement du modèle...")
try:
    if os.path.exists(LOCAL_MODEL_PATH):
        logger.info(f"Mode autonome détecté - Chargement depuis le dossier local : {LOCAL_MODEL_PATH}")
        model = mlflow.pyfunc.load_model(LOCAL_MODEL_PATH)
    else:
        logger.info("Mode développement détecté - Connexion au serveur MLflow distant")
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        model = mlflow.pyfunc.load_model("models:/Champion_LightGBM/2")
    
    logger.info("Modèle chargé avec succès.")
except Exception as e:
    logger.error(f"Impossible de charger le modèle : {e}")
    model = None

# 6. Logique de prédiction de l'API
def predict(input_json_str):
    if not model or not expected_features:
        return "Erreur : L'application n'est pas prête."
    
    try:
        input_dict = json.loads(input_json_str)
    except json.JSONDecodeError:
        return "Erreur : Le texte fourni n'est pas un JSON valide."
        
    ordered_values = []
    for feature_name in expected_features:
        if feature_name not in input_dict:
            return f"Erreur : Variable manquante : {feature_name}"
            
        value = input_dict[feature_name]
        
        if not isinstance(value, (int, float)):
            return f"Erreur : La variable {feature_name} doit être un nombre."
            
        if value < -1000000 or value > 100000000:
             return f"Erreur : La valeur de {feature_name} est hors des limites acceptées."
             
        ordered_values.append(value)
        
    try:
        input_data = np.array(ordered_values, dtype=np.float32).reshape(1, -1)
        prediction = model.predict(input_data)
        return str(float(prediction[0]))
    except Exception as e:
        logger.error(f"Erreur d'inférence : {e}")
        return "Erreur lors du calcul de la prédiction."

# 7. Interface Gradio
app = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=10, label="Données client (Format JSON)"),
    outputs=gr.Textbox(label="Prédiction du modèle"),
    title="ML Scoring API",
    description="Fournissez les variables requises au format JSON pour obtenir une prédiction."
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8000)