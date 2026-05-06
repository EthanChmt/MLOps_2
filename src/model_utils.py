import numpy as np
import os
from sklearn.metrics import confusion_matrix

# Chemin de base dynamique pour s'adapter à ton arborescence
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data():
    """
    Charge les matrices de données préparées.
    """
    path = os.path.join(BASE_DIR, 'data', 'processed')
    X = np.load(os.path.join(path, 'train_final.npy')) 
    y = np.load(os.path.join(path, 'train_labels.npy'))
    return X, y

def custom_business_cost(y_true, y_pred_proba, threshold=0.5, **kwargs):
    """
    Calcule le coût métier avec gestion des matrices de probabilités.
    """
    # SECURITÉ : Si Scikit-learn envoie deux colonnes, on ne garde que la proba du défaut (classe 1)
    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
        y_pred_proba = y_pred_proba[:, 1]

    # Transformation des probabilités en classes selon le seuil
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Extraction de la matrice de confusion
    # On utilise labels=[0,1] pour être sûr de l'ordre tn, fp, fn, tp même si une classe est absente
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Calcul du coût pondéré (FN coûte 10 fois plus que FP)
    total_cost = (fn * 10) + (fp * 1)
    return total_cost