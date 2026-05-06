import logging
import warnings
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.model_utils import custom_business_cost

# Configuration du logging pour le module
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

def pipeline_model(model_type):
    """
    Définit les pipelines et les grilles d'hyperparamètres.
    
    """
    if model_type == "Logistic_Regression":
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ])
        param_grid = { #Grille d'hyperparamètres pour la régression logistique
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__penalty': ['l2']
        }
        
    elif model_type == "XGBoost":
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', XGBClassifier(scale_pos_weight=11, random_state=42, eval_metric='logloss'))
        ])
        param_grid = {
            'classifier__max_depth': [3, 5, 7],
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1]
        }

    elif model_type == "LightGBM":
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', LGBMClassifier(class_weight='balanced', random_state=42, force_col_wise=True))
        ])
        param_grid = {
            'classifier__num_leaves': [20, 31, 50],
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1]
        }
    else:
        raise ValueError(f"Modèle {model_type} non supporté.")
    
    return pipeline, param_grid

def find_best_threshold(y_true, y_proba):
    """
    Identifie le seuil de probabilité qui minimise le coût métier.
    Retourne le meilleur seuil, le coût minimal, ainsi que les listes pour le graphique.
    """
    thresholds = np.linspace(0.1, 0.9, 81) # Seuils de 0.1 à 0.9 avec un pas de 0.01
    costs = [custom_business_cost(y_true, y_proba, threshold=t) for t in thresholds] # Calcul du coût pour chaque seuil
    
    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    min_cost = costs[best_idx]
    
    return best_threshold, min_cost, thresholds, costs