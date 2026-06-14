import os
import sys
import io
import json
import logging
import time
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import numpy as np

# --- 1. CONFIGURATION SYSTEME ---
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
os.chdir(root_path)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

base_dir = os.path.abspath(os.path.dirname(__file__))
features_path = os.path.join(base_dir, "expected_features.json")

monitoring_dir = os.path.join(root_path, "monitoring")
log_file = os.path.join(monitoring_dir, "logs.csv")

# --- 1.5 CONFIGURATION BASE DE DONNÉES ---
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    class PredictionLog(Base):
        __tablename__ = "prediction_logs"
        
        id = Column(Integer, primary_key=True, index=True)
        timestamp = Column(DateTime, default=datetime.utcnow)
        input_data = Column(JSON)
        prediction = Column(Float, nullable=True)
        status = Column(String)
        execution_time = Column(Float, nullable=True)

    Base.metadata.create_all(bind=engine)
else:
    logger.warning("DATABASE_URL non définie. Les logs BDD seront désactivés.")

# --- 2. INITIALISATION FASTAPI ---
app = FastAPI(
    title="ML Scoring API",
    description="API de scoring. Applique le feature engineering métier et formate les données pour l'inférence."
)

model = None
expected_features = []

# --- 3. LOGIQUE MÉTIER ---
def apply_feature_engineering(df):
    df = df.copy()

    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243)
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan})
        
    if 'DAYS_BIRTH' in df.columns:
        df['DAYS_BIRTH_YEARS'] = df['DAYS_BIRTH'] / -365

    if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        
    if 'AMT_CREDIT' in df.columns and 'AMT_ANNUITY' in df.columns:
        df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        
    if 'DAYS_EMPLOYED' in df.columns and 'DAYS_BIRTH' in df.columns:
        df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    df = pd.get_dummies(df)

    return df

def log_predictions(df: pd.DataFrame, predictions: list = None, status: str = "success", exec_time: float = None):
    if not DATABASE_URL:
        return

    try:
        db = SessionLocal()
        df_clean = df.replace({np.nan: None})
        data_dicts = df_clean.to_dict(orient="records")
        
        for i, row_data in enumerate(data_dicts):
            pred = predictions[i] if predictions and i < len(predictions) else None
            log_entry = PredictionLog(
                input_data=row_data,
                prediction=pred,
                status=status,
                execution_time=exec_time
            )
            db.add(log_entry)
            
        db.commit()
    except Exception as e:
        logger.error(f"Erreur BDD : {e}")
    finally:
        db.close()

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
        #model = joblib.load(model_path)
        model = ort.InferenceSession("model.onnx")
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
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Fichier CSV requis.")

    if model is None or not expected_features:
        raise HTTPException(status_code=503, detail="API indisponible.")

    try:
        start_time = time.time()
        contents = await file.read()
        
        df_brut = pd.read_csv(io.BytesIO(contents), sep=',')
        df_brut.columns = df_brut.columns.str.strip().str.replace('"', '').str.replace("'", "")
        
        colonnes_requises = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED"]
        colonnes_manquantes = [col for col in colonnes_requises if col not in df_brut.columns]
        
        if colonnes_manquantes:
            execution_duration = time.time() - start_time
            background_tasks.add_task(log_predictions, df_brut, None, "erreur_colonnes_manquantes", execution_duration)
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes manquantes détectées : {colonnes_manquantes}"
            )
            
        df_engineered = apply_feature_engineering(df_brut)
        df_final = df_engineered.reindex(columns=expected_features, fill_value=0)
        
        #predictions = model.predict(df_final).tolist()
        X = df_final.astype(np.float32).values
        input_name = model.get_inputs()[0].name
        predictions = model.run(None, {input_name: X})[0].tolist()

        execution_duration = time.time() - start_time
        background_tasks.add_task(log_predictions, df_brut, predictions, "success", execution_duration)
        
        return {"predictions": predictions}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")