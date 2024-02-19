"""
Módulo para la inferencia del modelo.

Este módulo contiene funciones para cargar un modelo entrenado 
y realizar inferencias en nuevos datos.
"""

import pandas as pd
import joblib

def load_model(model_path):
    """
    Carga un modelo entrenado desde un archivo utilizando joblib.

    Parámetros:
    - model_path (str): Ruta al archivo del modelo entrenado.

    Retorna:
    - model: Modelo cargado.
    """
    # Utiliza joblib para cargar el modelo entrenado desde el disco
    return joblib.load(model_path)

def load_inference_data(data_path):
    """
    Carga los datos de inferencia desde un archivo CSV.

    Parámetros:
    - data_path (str): Ruta al archivo CSV de los datos de inferencia.

    Retorna:
    - DataFrame: Datos de inferencia cargados como un DataFrame de pandas.
    """
    # Utiliza pandas para leer los datos de inferencia desde un archivo CSV
    return pd.read_csv(data_path)

def make_predictions(model, data):
    """
    Realiza predicciones utilizando el modelo proporcionado sobre los datos dados.

    Parámetros:
    - model: Modelo entrenado para hacer predicciones.
    - data (DataFrame): Datos sobre los cuales realizar predicciones.

    Retorna:
    - array: Predicciones generadas por el modelo.
    """
    # Utiliza el método predict del modelo para generar predicciones sobre los datos proporcionados
    return model.predict(data)
