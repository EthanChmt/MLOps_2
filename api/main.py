import os
import json
import joblib
import numpy as np
import gradio as gr
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

base_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(base_dir, "model.pkl")
features_path = os.path.join(base_dir, "expected_features.json")

model = None
expected_features = []

try:
    model = joblib.load(model_path)
    with open(features_path, 'r') as f:
        expected_features = json.load(f)
    logger.info("Modèle et signature des variables chargés.")
except Exception as e:
    logger.error(f"Erreur d'initialisation : {e}")

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
        ordered_values.append(input_dict[feature_name])
        
    try:
        input_data = np.array(ordered_values).reshape(1, -1)
        prediction = model.predict(input_data)
        return str(float(prediction[0]))
    except Exception as e:
        logger.error(f"Erreur d'inférence : {e}")
        return "Erreur lors du calcul de la prédiction."

app = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=10, label="Données client (Format JSON)"),
    outputs=gr.Textbox(label="Prédiction du modèle"),
    title="ML Scoring API",
    description="Fournissez les variables requises au format JSON pour obtenir une prédiction."
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8000)