
"""Módulo para la inferencia del modelo.

Este módulo contiene funciones para cargar un modelo entrenado 
y realizar inferencias en nuevos datos.
"""

# src/model_inference.py
import pandas as pd
import joblib

def load_model(model_path):
    # Cargar el modelo entrenado usando joblib
    return joblib.load(model_path)

def load_inference_data(data_path):
    # Cargar los datos para inferencia
    return pd.read_csv(data_path)

def make_predictions(model, data):
    # Realizar predicciones con el modelo
    return model.predict(data)
