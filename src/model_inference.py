"""
Módulo para la inferencia del modelo.

Este módulo contiene funciones para cargar un modelo entrenado 
y realizar inferencias en nuevos datos.
"""

import logging
import pandas as pd
import joblib
from src.utils import model_training_exception_handler

@model_training_exception_handler
def load_model(model_path):
    """
    Carga un modelo entrenado desde un archivo utilizando joblib.

    Parámetros:
    - model_path (str): Ruta al archivo del modelo entrenado.

    Retorna:
    - model: Modelo cargado.
    """
    # Utiliza joblib para cargar el modelo entrenado desde el disco
    logging.debug("Cargando el modelo desde la ruta: %s", model_path)
    model = joblib.load(model_path)
    logging.debug("Modelo cargado exitosamente.")
    return model

@model_training_exception_handler
def load_inference_data(data_path):
    """
    Carga los datos de inferencia desde un archivo CSV.

    Parámetros:
    - data_path (str): Ruta al archivo CSV de los datos de inferencia.

    Retorna:
    - DataFrame: Datos de inferencia cargados como un DataFrame de pandas.
    """
    # Utiliza pandas para leer los datos de inferencia desde un archivo CSV
    logging.debug("Cargando datos de inferencia desde la ruta: %s", data_path)
    data = pd.read_csv(data_path)
    logging.debug("Datos de inferencia cargados con %d filas y %d columnas.",
                  data.shape[0],
                  data.shape[1])
    return data

@model_training_exception_handler
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
    logging.debug("Realizando predicciones en los datos proporcionados.")
    predictions = model.predict(data)
    logging.debug("Predicciones realizadas exitosamente.")
    return predictions
