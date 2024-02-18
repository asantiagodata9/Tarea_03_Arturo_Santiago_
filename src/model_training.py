"""
Este módulo contiene funciones para cargar datos preparados, entrenar un modelo 
CatBoostRegressor y obtener características categóricas.

Funciones:
- load_prepared_data(path): Carga los datos preparados desde un archivo CSV.
- train_model(train_features, train_labels, cat_features, model_params, 
model_path='data/model/model.joblib'):
Entrena un modelo CatBoostRegressor con parámetros dados y lo guarda en disco.
- get_categorical_features(train_features): 
Obtiene las características categóricas de un DataFrame.
"""

import pandas as pd
from catboost import CatBoostRegressor
import joblib  # Importar joblib

def load_prepared_data(path):
    """
    Carga los datos preparados desde un archivo CSV.

    Parámetros:
    - path (str): Ruta del archivo CSV que contiene los datos preparados.

    Retorna:
    - pandas.DataFrame: DataFrame que contiene los datos preparados.
    """
    return pd.read_csv(path)

def train_model(train_features, train_labels, cat_features, model_params, model_path='data/model/model.joblib'):
    """
    Entrena un modelo CatBoostRegressor con parámetros dados y lo guarda en disco.

    Parámetros:
    - train_features (pandas.DataFrame): 
    DataFrame que contiene las características de entrenamiento.
    - train_labels (pandas.Series): 
    Series que contiene las etiquetas de entrenamiento.
    - cat_features (list): 
    Lista de nombres de características categóricas.
    - model_params (dict): 
    Diccionario de hiperparámetros para el modelo.
    - model_path (str): 
    Ruta donde se guardará el modelo.

    Retorna:
    - None
    """
    model = CatBoostRegressor(**model_params)
    model.fit(train_features, train_labels, cat_features=cat_features)
    joblib.dump(model, model_path)  # Guardar el modelo con joblib
    print("Model trained and saved at:", model_path)

def get_categorical_features(train_features):
    """
    Obtiene las características categóricas de un DataFrame.

    Parámetros:
    - train_features (pandas.DataFrame): DataFrame que contiene las características.

    Retorna:
    - list: Lista de nombres de características categóricas.
    """
    return [col for col in train_features.columns if train_features[col].dtype == 'object']
