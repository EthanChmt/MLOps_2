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
    description="API stricte de scoring. Applique le feature engineering métier et valide l'intégrité des données."
)

model = None
expected_features = []

# --- 3. LOGIQUE MÉTIER (Ton Feature Engineering) ---
def apply_feature_engineering(df):
    """
    Applique les transformations exactes issues de l'Analyse Exploratoire.
    """
    df = df.copy()

    # A. Traitement des anomalies
    if 'DAYS_EMPLOYED' in df.columns:
        # Création du flag d'anomalie
        df['DAYS_EMPLOYED_ANOM'] = df['DAYS_EMPLOYED'] == 365243
        # Remplacement de l'anomalie par NaN (géré plus tard par le SimpleImputer du modèle)
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan})

    # B. Création des nouvelles variables (Ratios métiers)
    if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        
    if 'AMT_CREDIT' in df.columns and 'AMT_ANNUITY' in df.columns:
        df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
        
    if 'DAYS_EMPLOYED' in df.columns and 'DAYS_BIRTH' in df.columns:
        df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    # C. Encodage des variables catégorielles (One-Hot Encoding)
    # pd.get_dummies va transformer le texte en colonnes numériques (0 et 1)
    df = pd.get_dummies(df)

    return df

# --- 4. CHARGEMENT AU DÉMARRAGE ---
@app.on_event("startup")
def startup_event():
    global model, expected_features
    logger.info("Démarrage de l'API...")
    
    # Chargement du JSON des 473 colonnes
    try:
        with open(features_path, 'r') as f:
            expected_features = json.load(f)
        logger.info(f"{len(expected_features)} variables métier attendues chargées.")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du JSON : {e}")

    # Chargement du Modèle (le pipeline Imputer + Scaler + LightGBM)
    try:
        # À MODIFIER avec ton repo Hugging Face
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
    return {"status": "ok", "message": "L'API est fonctionnelle et stricte."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Veuillez fournir un fichier CSV.")
    
    if model is None or not expected_features:
        raise HTTPException(status_code=503, detail="Le service de prédiction est indisponible.")

    try:
        # 1. Lecture du fichier brut
        contents = await file.read()
        df_brut = pd.read_csv(io.BytesIO(contents))
        
        # 2. Application de TON Feature Engineering
        df_engineered = apply_feature_engineering(df_brut)
        
        # 3. CONTRÔLE STRICT (L'API bloque si le résultat ne matche pas ton EDA)
        missing_cols = [col for col in expected_features if col not in df_engineered.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Erreur de format de données. Après transformation, il manque les colonnes suivantes : {missing_cols[:10]}..."
            )
            
        # 4. Filtrage et alignement parfait pour le modèle
        df_final = df_engineered[expected_features]
        
        # 5. Inférence
        predictions = model.predict(df_final)
        
        return {"predictions": predictions.tolist()}
    
    except HTTPException:
        raise # On laisse passer nos propres erreurs de validation (ex: les 400)
    except Exception as e:
        logger.error(f"Erreur d'inférence : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne lors du traitement : {str(e)}")