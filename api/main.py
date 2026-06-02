import os
import sys
import io
import json
import logging
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from huggingface_hub import hf_hub_download

# --- 1. CONFIGURATION SYSTEME ---
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
os.chdir(root_path)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

base_dir = os.path.abspath(os.path.dirname(__file__))
features_path = os.path.join(base_dir, "expected_features.json")

# --- 2. INITIALISATION FASTAPI ---
app = FastAPI(
    title="ML Scoring API",
    description="API de scoring. Applique le feature engineering métier et formate les données pour l'inférence."
)

model = None
expected_features = []

# --- 3. LOGIQUE MÉTIER ---
def apply_feature_engineering(df):
    """
    Applique les transformations exactes issues de l'Analyse Exploratoire.
    """
    df = df.copy()

    # Traitement des anomalies et âges
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243)
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan})
        
    if 'DAYS_BIRTH' in df.columns:
        df['DAYS_BIRTH_YEARS'] = df['DAYS_BIRTH'] / -365

    # Création des nouvelles variables métiers
    if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        
    if 'AMT_CREDIT' in df.columns and 'AMT_ANNUITY' in df.columns:
        df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        
    if 'DAYS_EMPLOYED' in df.columns and 'DAYS_BIRTH' in df.columns:
        df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    # Encodage des variables catégorielles
    df = pd.get_dummies(df)

    return df

# --- 4. CHARGEMENT AU DÉMARRAGE ---
@app.on_event("startup")
def startup_event():
    global model, expected_features
    logger.info("Démarrage de l'API...")
    
    try:
        with open(features_path, 'r') as f:
            expected_features = json.load(f)
        logger.info(f"{len(expected_features)} variables métier attendues chargées.")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du JSON : {e}")

    try:
        repo_id = os.getenv("HF_MODEL_REPO", "EthanChmt/scoring_api_OC")
        logger.info(f"Téléchargement du modèle depuis {repo_id}...")
        model_path = hf_hub_download(repo_id=repo_id, filename="model.pkl")
        model = joblib.load(model_path)
        logger.info("Modèle chargé en mémoire avec succès.")
    except Exception as e:
        logger.error(f"Erreur critique lors du chargement du modèle : {e}")

# --- 5. ROUTES DE L'API ---
@app.get("/health")
def health_check():
    if model is None or not expected_features:
        raise HTTPException(status_code=503, detail="L'API n'est pas prête.")
    return {"status": "ok", "message": "L'API est fonctionnelle."}

@app.post("/predict")
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Vérification du format du fichier
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Veuillez fournir un fichier CSV.")
    
    # Vérification de la disponibilité du modèle
    if model is None or not expected_features:
        raise HTTPException(status_code=503, detail="Le service de prédiction est indisponible.")

    try:
        # Lecture du fichier
        contents = await file.read()
        df_brut = pd.read_csv(io.BytesIO(contents))
        
        # --- 1. SÉCURITÉ : Vérification des colonnes brutes indispensables ---
        # Si une de ces colonnes manque, le ratio ne peut pas être calculé et le test pytest échoue.
        colonnes_requises = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED"]
        colonnes_manquantes = [col for col in colonnes_requises if col not in df_brut.columns]
        
        if colonnes_manquantes:
            raise HTTPException(
                status_code=400,
                detail=f"Fichier invalide. Il manque ces colonnes essentielles : {colonnes_manquantes}"
            )
            
        # --- 2. APPLICATION DU FEATURE ENGINEERING ---
        df_engineered = apply_feature_engineering(df_brut)
        
        # --- 3. ALIGNEMENT AUTOMATIQUE DES COLONNES ---
        df_final = df_engineered.reindex(columns=expected_features, fill_value=0)
        
        # --- 4. PRÉDICTION ---
        predictions = model.predict(df_final)
        
        return {"predictions": predictions.tolist()}
    
    except HTTPException:
        # Permet de renvoyer proprement notre erreur 400 sans qu'elle devienne une erreur 500
        raise
    except Exception as e:
        logger.error(f"Erreur d'inférence : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")