
"""Módulo para la inferencia del modelo.

Este módulo contiene funciones para cargar un modelo entrenado 
y realizar inferencias en nuevos datos.
"""

import pandas as pd
import joblib

def make_predictions(model_path, inference_data_path, predictions_path):
    """Realiza predicciones usando un modelo entrenado y guarda los resultados.
    
    Args:
        model_path (str): Ruta al modelo entrenado.
        inference_data_path (str): Ruta a los datos para inferencia.
        predictions_path (str): Ruta para guardar las predicciones.
    """
    model = joblib.load(model_path)
    inference_data = pd.read_csv(inference_data_path)
    predictions = model.predict(inference_data)
    pd.DataFrame(predictions, columns=['SalePrice']).to_csv(predictions_path, index=False)
