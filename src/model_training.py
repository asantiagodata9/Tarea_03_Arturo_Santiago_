
"""Módulo para el entrenamiento del modelo.

Este módulo contiene funciones para entrenar un modelo de regresión y guardar el modelo entrenado.
"""

from catboost import CatBoostRegressor
import joblib
import pandas as pd

def train_model(train_features_path, train_labels_path, cat_features, model_path='model.joblib'):
    """Entrena un modelo de regresión y lo guarda en el disco.
    
    Args:
        train_features_path (str): Ruta al archivo CSV de características de entrenamiento.
        train_labels_path (str): Ruta al archivo CSV de etiquetas de entrenamiento.
        cat_features (list): Lista de nombres de características categóricas.
        model_path (str): Ruta para guardar el modelo entrenado.
    """
    train_features = pd.read_csv(train_features_path)
    train_labels = pd.read_csv(train_labels_path)
    model = CatBoostRegressor(iterations=1000,
                              learning_rate=0.1,
                              depth=6,
                              loss_function='RMSE',
                              verbose=200)
    model.fit(train_features, train_labels, cat_features=cat_features)
    joblib.dump(model, model_path)
